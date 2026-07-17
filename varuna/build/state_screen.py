"""Tier A: state-wide recharge-opportunity screen (V3 "Bhujal").

The property that makes state scale free-tier-feasible: **this module never downloads a
raster.** All pixel work happens server-side in Earth Engine via `reduceRegions`; the only
thing that crosses the wire is a table of a few hundred rows (~200 KB). Everything downstream
of the one EE-touching function is pure Python on that table, so scoring, normalization and
the sensitivity analysis are testable with canned rows — no EE, no network.

Recharge Opportunity Index, per admin unit:

    ROI = (w_need*need + w_water*water + w_soil*soil + w_land*land) / sum(w)

    need  : CGWB 2024 stage-of-extraction category, sharpened by the GLDAS (GRACE-assimilated)
            groundwater-storage trend. This is the authoritative "very little groundwater"
            signal and it DOMINATES by design (w_need ~ 0.45): a block with perfect soil but
            no depletion is a place where recharge is *possible*, not *needed*.
    water : runoff concentration — log1p(HydroSHEDS flow accumulation p90), state-normalized.
    soil  : Cosby Ksat from per-block mean SoilGrids sand/clay, capped (sand that drains to
            nowhere useful is not a win), state-normalized.
    land  : pervious fraction (WorldCover, canonical PERVIOUS set). Already a physical 0-1
            fraction, so it is NOT percentile-stretched — stretching a compressed range
            manufactures contrast that is not in the data.

All normalization is STATE-WIDE across blocks, never within-block: RSI's within-AOI 2-98
stretch is exactly how a constant field became structured noise (see recharge.py history).
A state-wide constant input carries no ranking information, so `_norm` returns zeros for it
and says so in `screen_notes` instead of amplifying float dust.

Honesty spine: the screen is a screen. GLDAS is 27.8 km and GRACE-assimilated; CGWB is
annual and per-unit. Tier A may say "this unit is depleted AND has runoff AND has pervious
ground". It cannot say "build here" — only a Tier B twin, re-simulated with a metered
aquifer, earns m³ claims. Every payload carries that note plus the water-quality caveat.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import logging
import os

import numpy as np

from ..io import save_json
from .landcover import IMPERVIOUS, PERVIOUS
from .recharge import QUALITY_CAVEAT, cosby_ksat

log = logging.getLogger("varuna.build.state_screen")

# ---------------------------------------------------------------- hyperparameters (tweak me)

# GEC-2015 categorization by stage of extraction (%): the official CGWB mapping.
# (The full methodology also validates against water-level trends; the national compilation's
# published category tracks these stage bands, which is what we reproduce from the CSV.)
CATEGORY_BANDS = (("Over-exploited", 100.0), ("Critical", 90.0), ("Semi-critical", 70.0),
                  ("Safe", -np.inf))

# How much "need" each category asserts. Safe is 0.15, not 0.0 — a Safe block can still hold
# a depleting trend worth watching, and a hard zero would erase the trend term entirely.
CGWB_CATEGORY_SCORE = {"Over-exploited": 1.0, "Critical": 0.75, "Semi-critical": 0.5,
                       "Safe": 0.15}

# ROI weights. Sum is normalized away, so tweak freely; need must stay dominant.
ROI_WEIGHTS = {"need": 0.45, "water": 0.25, "soil": 0.15, "land": 0.15}

# The GLDAS depletion trend can raise `need` by at most this factor above the bare category
# score (trend corroborates and sharpens the annual CGWB category; it must not override it).
DEPLETION_BONUS = 0.5

# Cosby Ksat cap (mm/hr) before normalization: beyond this, extra permeability stops being
# extra opportunity (water drains past the root/storage zone instead of recharging usefully).
KSAT_CAP_MM_HR = 50.0

# State-wide normalization percentiles. 5-95 rather than min-max so one freak block does not
# compress everyone else onto [0, 0.1]; with ~31 districts these are already coarse.
NORM_PCT = (5.0, 95.0)

# Sensitivity analysis: n weight perturbations, each weight jittered by U(1±jitter), renormed.
SENSITIVITY_N = 200
SENSITIVITY_JITTER = 0.30

# ------------------------------------------------------------------------- CGWB table loading

# Modern Karnataka district names (CGWB 2024 CSV) -> FAO GAUL 2015 names. GAUL predates the
# 2014+ renamings wave. Names not listed pass through `_canon` + fuzzy matching, and every
# fuzzy decision is recorded in the join report — nothing matches silently.
GAUL_ALIASES = {
    "bengaluru urban": "bangalore urban",
    "bengaluru rural": "bangalore rural",
    "belagavi": "belgaum",
    "ballari": "bellary",
    "vijayapura": "bijapur",
    "kalburgi": "gulbarga",
    "kalaburagi": "gulbarga",
    "mysuru": "mysore",
    "shivamogga": "shimoga",
    "tumakuru": "tumkur",
    "chikkamagaluru": "chikmagalur",
    "chamarajanagara": "chamarajanagar",
    "chikkaballapura": "chikballapur",
    "kolara": "kolar",
}

# Districts created after GAUL 2015, merged into the parent whose polygon still contains them.
# The merge recomputes stage volumetrically for the combined polygon but ships every member's
# own stage in `members`, so e.g. Vijayanagara's 88% is not hidden inside Ballari's 27%.
POST_GAUL_SPLITS = {"Vijayanagara": "Ballari"}


def _canon(name):
    """Canonical join key: lowercase, parentheses/punctuation stripped, spaces collapsed."""
    s = str(name).lower().replace("(", " ").replace(")", " ").replace(".", " ").replace("-", " ")
    return " ".join(s.split())


def category_from_stage(stage_pct):
    """GEC-2015 stage-of-extraction (%) -> category string."""
    for cat, floor in CATEGORY_BANDS:
        if float(stage_pct) > floor:
            return cat
    return "Safe"


def load_cgwb(path):
    """data/cgwb_karnataka_2024_district.csv -> tidy DataFrame (units: hectare-metres).

    Source: CGWB 'National Compilation on Dynamic Ground Water Resources of India 2024'
    (district table via data.opencity.in mirror — see data/README_cgwb.md for provenance and
    the taluk-level upgrade path). Total rows are dropped; category derives from stage.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    df = df.rename(columns={
        "Name of District": "district",
        "Total annual groundwater recharge": "recharge_ham",
        "Annual Extractable Groundwater Resource": "extractable_ham",
        "Total Annual Extraction": "extraction_ham",
        "Net GW availability for future": "future_ham",
        "Stage of GW extraction (%)": "stage_pct",
    })
    df = df[~df["district"].astype(str).str.startswith("Total")].copy()
    keep = ["district", "recharge_ham", "extractable_ham", "extraction_ham", "future_ham",
            "stage_pct"]
    df = df[keep]
    for c in keep[1:]:
        df[c] = df[c].astype(float)
    df["category"] = [category_from_stage(s) for s in df["stage_pct"]]
    df["category_score"] = df["category"].map(CGWB_CATEGORY_SCORE)
    return df.reset_index(drop=True)


def join_units(cgwb, polygon_names):
    """Match CGWB rows to polygon unit names. Returns (joined_df, report).

    Three mechanisms, every one of them recorded in the report (nothing matches silently):
      1. exact match on canonical names, via GAUL_ALIASES for known renames;
      2. POST_GAUL_SPLITS: districts newer than the polygon vintage merge into their parent —
         volumetric columns sum, stage recomputes, members keep their own stages;
      3. difflib fuzzy match (cutoff 0.75) for spelling drift (Davanagere/Davangere...).
    """
    import pandas as pd
    cgwb = cgwb.copy()
    poly_by_canon = {_canon(p): p for p in polygon_names}
    report = {"exact": [], "alias": [], "fuzzy": [], "merged": [],
              "unmatched_cgwb": [], "unmatched_polygons": []}

    # --- step 2 first: fold post-split districts into their parents
    for child, parent in POST_GAUL_SPLITS.items():
        crow = cgwb[cgwb["district"] == child]
        prow = cgwb[cgwb["district"] == parent]
        if len(crow) and len(prow):
            merged = prow.iloc[0].copy()
            members = [{"district": r["district"], "stage_pct": round(r["stage_pct"], 1),
                        "category": r["category"]} for _, r in
                       pd.concat([prow, crow]).iterrows()]
            for c in ["recharge_ham", "extractable_ham", "extraction_ham", "future_ham"]:
                merged[c] = prow.iloc[0][c] + crow.iloc[0][c]
            merged["stage_pct"] = 100.0 * merged["extraction_ham"] / merged["extractable_ham"]
            merged["category"] = category_from_stage(merged["stage_pct"])
            merged["category_score"] = CGWB_CATEGORY_SCORE[merged["category"]]
            cgwb = cgwb[~cgwb["district"].isin([child, parent])]
            merged["members"] = members
            cgwb = pd.concat([cgwb, merged.to_frame().T], ignore_index=True)
            report["merged"].append({"into": parent, "members": [m["district"] for m in members]})

    # --- steps 1 + 3: name matching
    unit_col, matched_polys = [], set()
    for _, row in cgwb.iterrows():
        canon = _canon(row["district"])
        canon = GAUL_ALIASES.get(canon, canon)
        if canon in poly_by_canon:
            unit_col.append(poly_by_canon[canon])
            matched_polys.add(poly_by_canon[canon])
            kind = "alias" if _canon(row["district"]) in GAUL_ALIASES else "exact"
            report[kind].append({"cgwb": row["district"], "polygon": poly_by_canon[canon]})
            continue
        close = difflib.get_close_matches(canon, list(poly_by_canon), n=1, cutoff=0.75)
        if close:
            unit_col.append(poly_by_canon[close[0]])
            matched_polys.add(poly_by_canon[close[0]])
            report["fuzzy"].append({"cgwb": row["district"], "polygon": poly_by_canon[close[0]]})
        else:
            unit_col.append(None)
            report["unmatched_cgwb"].append(row["district"])
    cgwb["unit"] = unit_col
    report["unmatched_polygons"] = sorted(set(polygon_names) - matched_polys)
    report["coverage"] = round(float((cgwb["unit"].notna()).mean()), 3)
    joined = cgwb[cgwb["unit"].notna()].reset_index(drop=True)
    return joined, report


# --------------------------------------------------------------- the ONE EE-touching function

def fetch_unit_table(fc, name_prop="ADM2_NAME", start="2003-02-01", wc_scale=100):
    """reduceRegions over the five state-wide layers -> list of plain dicts (tables, no rasters).

    Layers (all free):
      GLDAS-2.2 CLSM GWS_tavg (GRACE-assimilated groundwater storage, 27.8 km): monthly means
        -> per-pixel linear trend mm/yr (ee.Reducer.linearFit over [t_years, GWS]) -> unit mean.
      SoilGrids sand/clay g/kg, mean of 0-5/5-15/15-30 cm (same construction as
        recharge.download_soil) -> unit mean, Cosby applied later in Python.
      ESA WorldCover v200: canonical PERVIOUS / IMPERVIOUS fractions (sampled at `wc_scale` m —
        a screen does not need 10 m; 100 m keeps EE happy over 191,000 km²).
      WWF HydroSHEDS 15ACC flow accumulation: p90 + max per unit (where runoff concentrates).

    This function is the module's only EE dependency; cache its output (run_state_screen.py
    --cache) and every downstream number is reproducible + tweakable offline.
    """
    import ee

    def _reduce(img, reducer, scale, keys):
        feats = img.reduceRegions(collection=fc, reducer=reducer, scale=scale,
                                  tileScale=4).getInfo()["features"]
        out = {}
        for f in feats:
            p = f["properties"]
            out[p[name_prop]] = {k: p.get(k) for k in keys}
        return out

    # GLDAS monthly means with a fractional-year time band, then linearFit(t, gws)
    coll = ee.ImageCollection("NASA/GLDAS/V022/CLSM/G025/DA1D").select("GWS_tavg")
    start_d = ee.Date(start)
    end_d = ee.Date(ee.ImageCollection("NASA/GLDAS/V022/CLSM/G025/DA1D")
                    .limit(1, "system:time_start", False).first().get("system:time_start"))
    n_months = end_d.difference(start_d, "month").floor()

    def month_img(m):
        m0 = start_d.advance(ee.Number(m), "month")
        t = ee.Image.constant(m0.difference(start_d, "year")).float().rename("t")
        return coll.filterDate(m0, m0.advance(1, "month")).mean().rename("gws").addBands(t)

    monthly = ee.ImageCollection(ee.List.sequence(0, n_months.subtract(1)).map(month_img))
    fit = monthly.select(["t", "gws"]).reduce(ee.Reducer.linearFit())    # scale = mm per year
    gldas = _reduce(fit.select("scale").rename("trend"), ee.Reducer.mean(), 27830, ["mean"])

    sand = ee.Image("projects/soilgrids-isric/sand_mean").select(
        ["sand_0-5cm_mean", "sand_5-15cm_mean", "sand_15-30cm_mean"]).reduce(ee.Reducer.mean())
    clay = ee.Image("projects/soilgrids-isric/clay_mean").select(
        ["clay_0-5cm_mean", "clay_5-15cm_mean", "clay_15-30cm_mean"]).reduce(ee.Reducer.mean())
    sand_t = _reduce(sand, ee.Reducer.mean(), 250, ["mean"])
    clay_t = _reduce(clay, ee.Reducer.mean(), 250, ["mean"])

    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    perv_img = wc.remap(sorted(PERVIOUS), [1] * len(PERVIOUS), 0)
    built_img = wc.remap(sorted(IMPERVIOUS), [1] * len(IMPERVIOUS), 0)
    perv = _reduce(perv_img, ee.Reducer.mean(), wc_scale, ["mean"])
    built = _reduce(built_img, ee.Reducer.mean(), wc_scale, ["mean"])

    acc = ee.Image("WWF/HydroSHEDS/15ACC")
    acc_t = _reduce(acc, ee.Reducer.percentile([90]).combine(ee.Reducer.max(),
                                                             sharedInputs=True), 500,
                    ["p90", "max"])

    rows = []
    for unit in gldas:
        rows.append({
            "unit": unit,
            "gws_trend_mm_yr": gldas[unit]["mean"],
            "sand_gkg": sand_t.get(unit, {}).get("mean"),
            "clay_gkg": clay_t.get(unit, {}).get("mean"),
            "pervious_frac": perv.get(unit, {}).get("mean"),
            "built_frac": built.get(unit, {}).get("mean"),
            "acc_p90": acc_t.get(unit, {}).get("p90"),
            "acc_max": acc_t.get(unit, {}).get("max"),
        })
    return rows


def karnataka_units(admin_asset=None, admin_level="district", name_prop=None):
    """The unit polygons. Default: FAO/GAUL/2015/level2 filtered to Karnataka (DISTRICTS, ~30).

    Pass a custom `admin_asset` (uploaded taluk polygons) + its `name_prop` once the boundary
    spike (scripts/spike_boundaries.py) shows join coverage worth claiming taluk precision.
    """
    import ee
    if admin_asset:
        return ee.FeatureCollection(admin_asset), name_prop or "NAME", admin_level
    fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.eq("ADM1_NAME", "Karnataka"))
    return fc, "ADM2_NAME", "district"


# ----------------------------------------------------------------------- pure-Python scoring

def _norm(a, pct=NORM_PCT, notes=None, name=""):
    """State-wide percentile normalization with a constant-input guard.

    A column that is (near-)constant across the state carries zero ranking information; the
    within-AOI stretch in recharge.py amplified exactly such a column into structured noise.
    Here it normalizes to all-zeros and the fact is recorded in `screen_notes`.
    """
    a = np.asarray(a, dtype="float64")
    lo, hi = np.nanpercentile(a, pct[0]), np.nanpercentile(a, pct[1])
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-9:
        if notes is not None:
            notes.append(f"input '{name}' is state-wide constant — contributes nothing "
                         "to the ranking (not stretched into noise)")
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def compute_terms(df, ksat_cap=KSAT_CAP_MM_HR, norm_pct=NORM_PCT):
    """Raw joined table -> the four 0-1 term columns (+ derived ksat). Pure numpy, no EE.

    need  = category_score * (1 + DEPLETION_BONUS * depletion) / (1 + DEPLETION_BONUS)
            where depletion = state-normalized max(0, -gws_trend): only a *falling* GRACE
            trend sharpens need, and it can never lift a Safe block above a Critical one
            (max need for Safe = 0.15, min for Critical = 0.75/(1.5) = 0.5).
    water = state-normalized log1p(flow-accumulation p90).
    soil  = state-normalized min(cosby_ksat, ksat_cap).
    land  = pervious fraction, used raw (already physical and comparable).
    """
    notes = []
    df = df.copy()
    df["ksat_mm_hr"] = cosby_ksat(np.asarray(df["sand_gkg"], dtype="float64") / 10.0,
                                  np.asarray(df["clay_gkg"], dtype="float64") / 10.0)
    depletion = _norm(np.maximum(0.0, -np.asarray(df["gws_trend_mm_yr"], dtype="float64")),
                      norm_pct, notes, "gws depletion trend")
    df["term_need"] = (np.asarray(df["category_score"], dtype="float64")
                       * (1.0 + DEPLETION_BONUS * depletion) / (1.0 + DEPLETION_BONUS))
    df["term_water"] = _norm(np.log1p(np.asarray(df["acc_p90"], dtype="float64")),
                             norm_pct, notes, "flow accumulation p90")
    df["term_soil"] = _norm(np.minimum(np.asarray(df["ksat_mm_hr"], dtype="float64"), ksat_cap),
                            norm_pct, notes, "cosby ksat")
    df["term_land"] = np.asarray(df["pervious_frac"], dtype="float64")
    return df, notes


def score_from_terms(df, weights=None):
    """Terms + weights -> ROI column + dense rank (1 = best). Weight sum normalizes away."""
    w = dict(ROI_WEIGHTS, **(weights or {}))
    total = sum(w.values())
    roi = sum(w[k] * np.asarray(df[f"term_{k}"], dtype="float64") for k in w) / total
    df = df.copy()
    df["roi"] = roi
    df["rank"] = (-roi).argsort().argsort() + 1
    return df


def sensitivity_analysis(df, weights=None, n=SENSITIVITY_N, jitter=SENSITIVITY_JITTER, seed=0):
    """Re-rank under multiplicative weight perturbations U(1±jitter); report rank bands.

    The direct descendant of the paper's DEM-ensemble finding, one level up: a ranking that
    dissolves under a plausible perturbation should not drive a budget. Per unit: median /
    p5 / p95 rank and the fraction of perturbations that keep it in the top 10. `stable_top10`
    lists units in the top 10 in >=80% of perturbations — reported even if unflattering.
    """
    w0 = dict(ROI_WEIGHTS, **(weights or {}))
    rng = np.random.default_rng(seed)
    ranks = np.empty((n, len(df)), dtype="int64")
    for i in range(n):
        w = {k: v * rng.uniform(1.0 - jitter, 1.0 + jitter) for k, v in w0.items()}
        ranks[i] = np.asarray(score_from_terms(df, w)["rank"])
    out = df.copy()
    out["rank_median"] = np.median(ranks, axis=0).astype(int)
    out["rank_p5"] = np.percentile(ranks, 5, axis=0).astype(int)
    out["rank_p95"] = np.percentile(ranks, 95, axis=0).astype(int)
    out["top10_freq"] = (ranks <= 10).mean(axis=0).round(3)
    stable = out[out["top10_freq"] >= 0.8].sort_values("rank")["unit"].tolist()
    summary = {"n_perturbations": n, "jitter": jitter,
               "stable_top10": stable,
               "note": ("units in the top 10 under >=80% of weight perturbations; if this "
                        "list is much shorter than 10, the ranking is weight-sensitive and "
                        "must not drive a budget on its own")}
    return out, summary


# --------------------------------------------------------------------------------- pipeline

SCREEN_NOTE = (
    "A screen, not a siting. Inputs: CGWB 2024 stage-of-extraction (annual, per admin unit), "
    "GLDAS-2.2/GRACE-assimilated storage trend (27.8 km), HydroSHEDS flow accumulation, "
    "SoilGrids Cosby Ksat, WorldCover pervious fraction. It can say a unit is depleted AND "
    "has concentrated runoff AND pervious ground; it cannot say 'build here'. Only a Tier B "
    "re-simulated 60 m twin earns m³ claims, and within-block siting still carries the "
    "paper's DEM-ensemble uncertainty."
)


def score_blocks(rows, cgwb, weights=None, ksat_cap=KSAT_CAP_MM_HR, norm_pct=NORM_PCT,
                 n_perturb=SENSITIVITY_N, jitter=SENSITIVITY_JITTER):
    """Canned EE rows + CGWB table -> fully scored + sensitivity-banded DataFrame (no EE).

    This is the function tests exercise; `run()` is just EE fetch + this + JSON writing.
    Returns (scored_df, join_report, screen_notes, sensitivity_summary).
    """
    import pandas as pd
    raw = pd.DataFrame(rows)
    joined, report = join_units(cgwb, raw["unit"].tolist())
    df = raw.merge(joined, on="unit", how="inner")
    if len(df) < len(raw):
        report.setdefault("dropped_polygons", sorted(set(raw["unit"]) - set(df["unit"])))
    df, notes = compute_terms(df, ksat_cap=ksat_cap, norm_pct=norm_pct)
    df = score_from_terms(df, weights)
    df, sens = sensitivity_analysis(df, weights, n=n_perturb, jitter=jitter)
    return df.sort_values("rank").reset_index(drop=True), report, notes, sens


def run(cgwb_csv="data/cgwb_karnataka_2024_district.csv", out_dir="artifacts/karnataka",
        project_id=None, admin_asset=None, admin_level="district", name_prop=None,
        weights=None, raw_cache=None, **hyper):
    """Full Tier A pipeline. EE only for `fetch_unit_table` (or none, with raw_cache)."""
    from ..ee_auth import init_ee
    os.makedirs(out_dir, exist_ok=True)
    raw_path = raw_cache or os.path.join(out_dir, "state_screen_raw.json")
    if os.path.exists(raw_path):
        import json
        with open(raw_path) as f:
            cached = json.load(f)
        rows, admin_level = cached["rows"], cached.get("admin_level", admin_level)
        log.info("state screen: reusing cached EE table %s (%d rows) — delete it to refetch",
                 raw_path, len(rows))
    else:
        init_ee(project_id)
        fc, name_prop, admin_level = karnataka_units(admin_asset, admin_level, name_prop)
        rows = fetch_unit_table(fc, name_prop=name_prop)
        save_json(raw_path, {"rows": rows, "admin_level": admin_level,
                             "fetched": _dt.datetime.now(_dt.timezone.utc).isoformat()})
    cgwb = load_cgwb(cgwb_csv)
    df, report, notes, sens = score_blocks(rows, cgwb, weights=weights, **hyper)

    blocks = df.drop(columns=[c for c in ["members"] if c in df], errors="ignore")
    payload = {
        "schema_version": 1,
        "state": "karnataka",
        "admin_level": admin_level,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "weights": dict(ROI_WEIGHTS, **(weights or {})),
        "hyperparams": {"ksat_cap_mm_hr": hyper.get("ksat_cap", KSAT_CAP_MM_HR),
                        "norm_pct": list(hyper.get("norm_pct", NORM_PCT)),
                        "depletion_bonus": DEPLETION_BONUS,
                        "category_score": CGWB_CATEGORY_SCORE},
        "note": SCREEN_NOTE,
        "caveat": QUALITY_CAVEAT,
        "join_report": report,
        "screen_notes": notes,
        "sensitivity": sens,
        "blocks": blocks.to_dict(orient="records"),
    }
    # units whose CGWB rows were merged keep their member stages visible
    for rec, (_, row) in zip(payload["blocks"], df.iterrows()):
        if "members" in df.columns and isinstance(row.get("members"), list):
            rec["members"] = row["members"]
    out_path = os.path.join(out_dir, "state_screen.json")
    save_json(out_path, payload)
    log.info("state screen: %d units -> %s (admin_level=%s, join coverage %.0f%%)",
             len(df), out_path, admin_level, 100 * report["coverage"])
    return payload
