"""Tier A state-screen tests on canned tables — no Earth Engine, no network, no rasters.

The screen's scoring pipeline is pure Python downstream of `fetch_unit_table`, so everything
that decides a ranking (category bands, need dominance, state-wide normalization, the join,
the sensitivity bands) is pinned here with synthetic 3-block fixtures.
"""
import os

import numpy as np
import pandas as pd
import pytest

from varuna.build import state_screen as ss

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                        "cgwb_karnataka_2024_district.csv")


def _mk_cgwb(rows):
    """rows: (district, extractable_ham, extraction_ham). Other columns derived."""
    df = pd.DataFrame(rows, columns=["district", "extractable_ham", "extraction_ham"])
    df["recharge_ham"] = df["extractable_ham"] * 1.1
    df["future_ham"] = (df["extractable_ham"] - df["extraction_ham"]).clip(lower=0)
    df["stage_pct"] = 100.0 * df["extraction_ham"] / df["extractable_ham"]
    df["category"] = [ss.category_from_stage(s) for s in df["stage_pct"]]
    df["category_score"] = df["category"].map(ss.CGWB_CATEGORY_SCORE)
    return df


def _mk_rows():
    """Three canned 'EE table' units: A over-exploited/adequate, B safe/perfect, C middling."""
    return [
        dict(unit="A", gws_trend_mm_yr=-5.0, sand_gkg=300, clay_gkg=300,
             pervious_frac=0.50, built_frac=0.30, acc_p90=1000.0, acc_max=9000.0),
        dict(unit="B", gws_trend_mm_yr=+2.0, sand_gkg=900, clay_gkg=50,
             pervious_frac=0.95, built_frac=0.01, acc_p90=5000.0, acc_max=90000.0),
        dict(unit="C", gws_trend_mm_yr=-1.0, sand_gkg=500, clay_gkg=200,
             pervious_frac=0.70, built_frac=0.10, acc_p90=100.0, acc_max=800.0),
    ]


_CGWB_ABC = [("A", 100.0, 150.0),   # stage 150% -> Over-exploited
             ("B", 100.0, 30.0),    # stage 30%  -> Safe
             ("C", 100.0, 80.0)]    # stage 80%  -> Semi-critical


def test_category_bands():
    """GEC-2015 bands with strict '>' floors: 100.0 is Critical, not Over-exploited."""
    assert ss.category_from_stage(150) == "Over-exploited"
    assert ss.category_from_stage(100.0) == "Critical"
    assert ss.category_from_stage(95) == "Critical"
    assert ss.category_from_stage(90.0) == "Semi-critical"
    assert ss.category_from_stage(75) == "Semi-critical"
    assert ss.category_from_stage(70.0) == "Safe"
    assert ss.category_from_stage(10) == "Safe"


def test_need_dominates():
    """A Safe block with perfect soil/water/land must not outrank an Over-exploited block
    with merely adequate ground — recharge where possible is not recharge where needed."""
    df, report, notes, sens = ss.score_blocks(_mk_rows(), _mk_cgwb(_CGWB_ABC))
    assert report["coverage"] == 1.0
    top = df.iloc[0]
    assert top["unit"] == "A" and top["category"] == "Over-exploited"
    roi = dict(zip(df["unit"], df["roi"]))
    assert roi["A"] > roi["B"]


def test_constant_input_contributes_nothing():
    """The anti-RSI property: a state-wide constant column must normalize to zeros and be
    reported, not percentile-stretched into structured noise."""
    rows = _mk_rows()
    for r in rows:
        r["acc_p90"] = 777.0
    df, _, notes, _ = ss.score_blocks(rows, _mk_cgwb(_CGWB_ABC))
    assert np.allclose(df["term_water"], 0.0)
    assert any("flow accumulation" in n for n in notes)


def test_sensitivity_bands_populated():
    df, _, _, sens = ss.score_blocks(_mk_rows(), _mk_cgwb(_CGWB_ABC),
                                     n_perturb=50, jitter=0.3)
    assert {"rank_median", "rank_p5", "rank_p95", "top10_freq"} <= set(df.columns)
    assert (df["rank_p5"] <= df["rank_median"]).all()
    assert (df["rank_median"] <= df["rank_p95"]).all()
    assert sens["stable_top10"]                       # 3 units: all trivially in top 10
    assert (df["top10_freq"] == 1.0).all()


def test_sensitivity_zero_jitter_is_deterministic():
    df, _, _, _ = ss.score_blocks(_mk_rows(), _mk_cgwb(_CGWB_ABC), n_perturb=5, jitter=0.0)
    assert (df["rank_p5"] == df["rank_p95"]).all()
    assert (df["rank_p5"] == df["rank"]).all()


def test_join_alias_and_post_split_merge():
    """Vijayanagara (post-GAUL split) folds into Ballari: volumes sum, stage recomputes,
    and the members' own stages stay visible — dilution must not hide an 88% member."""
    cgwb = _mk_cgwb([("Ballari", 100.0, 20.0),        # Safe on its own
                     ("Vijayanagara", 100.0, 90.0),   # Semi-critical on its own
                     ("Shivamogga", 100.0, 50.0)])
    joined, report = ss.join_units(cgwb, ["Bellary", "Shimoga"])
    assert report["coverage"] == 1.0
    assert report["merged"] == [{"into": "Ballari", "members": ["Ballari", "Vijayanagara"]}]
    merged = joined[joined["unit"] == "Bellary"].iloc[0]
    assert merged["stage_pct"] == pytest.approx(55.0)         # (20+90)/(100+100)
    stages = {m["district"]: m["stage_pct"] for m in merged["members"]}
    assert stages == {"Ballari": 20.0, "Vijayanagara": 90.0}
    assert any(a["cgwb"] == "Shivamogga" and a["polygon"] == "Shimoga"
               for a in report["alias"])


def test_join_reports_unmatched_instead_of_guessing():
    cgwb = _mk_cgwb([("Atlantis", 100.0, 50.0)])
    joined, report = ss.join_units(cgwb, ["Bellary", "Shimoga"])
    assert len(joined) == 0
    assert report["unmatched_cgwb"] == ["Atlantis"]
    assert set(report["unmatched_polygons"]) == {"Bellary", "Shimoga"}


def test_load_cgwb_real_csv_reality_check():
    """Plan §Verification 7 at district level: Kolar, Chikkaballapur and Bengaluru Urban are
    long-standing over-exploited units. If the authoritative table disagrees, stop and look."""
    df = ss.load_cgwb(DATA_CSV)
    assert len(df) == 31                      # 31 districts, Total rows dropped
    cat = dict(zip(df["district"], df["category"]))
    assert cat["Kolara"] == "Over-exploited"
    assert cat["Chikkaballapura"] == "Over-exploited"
    assert cat["Bengaluru (Urban)"] == "Over-exploited"
    assert df["stage_pct"].between(0, 400).all()


# The actual FAO/GAUL/2015/level2 Karnataka frame, fetched live 2026-07-17: the PRE-2007
# 27 districts, GAUL's own spellings. Pinning it keeps the alias/split tables honest offline.
GAUL_KARNATAKA_2015 = [
    "Bagalkot", "Bangalore Rural", "Bangalore Urban", "Belgaum", "Bellary", "Bidar",
    "Bijapur", "Chamrajnagar", "Chikmagalur", "Chitradurga", "Dakshin Kannad", "Davanagere",
    "Dharwad", "Gadag", "Gulbarga", "Hassan", "Haveri", "Kodagu", "Kolar", "Koppal",
    "Mandya", "Mysore", "Raichur", "Shimoga", "Tumkur", "Udupi", "Uttar Kannand",
]


def test_full_cgwb_gaul_join_coverage():
    """All 31 CGWB 2024 districts must resolve against the real GAUL 27-name frame — via
    alias, fuzzy, or a post-split merge — with nothing silently dropped either way."""
    cgwb = ss.load_cgwb(DATA_CSV)
    joined, report = ss.join_units(cgwb, GAUL_KARNATAKA_2015)
    assert report["coverage"] == 1.0
    assert report["unmatched_cgwb"] == []
    assert report["unmatched_polygons"] == []
    assert len(joined) == 27
    merged_into = {m["into"] for m in report["merged"]}
    assert merged_into == {"Ballari", "Kolara", "Bengaluru (Rural)", "Kalburgi"}
    # Chikkaballapur's 164% must stay visible inside the merged Kolar unit
    kolar = joined[joined["unit"] == "Kolar"].iloc[0]
    stages = {m["district"]: m["stage_pct"] for m in kolar["members"]}
    assert stages["Chikkaballapura"] == pytest.approx(164.3, abs=0.1)
    assert kolar["category"] == "Over-exploited"


def test_cosby_ksat_orders_soils():
    """Sanity: sandy soil conducts far more than clayey; the formula is Cosby 1984."""
    from varuna.build.recharge import cosby_ksat
    sandy, clayey = cosby_ksat(80.0, 5.0), cosby_ksat(10.0, 60.0)
    assert sandy > 10 * clayey


def test_landcover_sets_are_consistent():
    """One definition, three consumers: the twin's F_TABLE gives every NO_RECHARGE class a
    zero rate, waterbalance's dashboard group equals the canonical set, sets are disjoint."""
    from varuna.build.landcover import IMPERVIOUS, NO_RECHARGE, PERVIOUS
    assert not (PERVIOUS & NO_RECHARGE) and not (PERVIOUS & IMPERVIOUS)
    from varuna.serve.waterbalance import CLASS_GROUPS
    assert CLASS_GROUPS["water_wetland"] == tuple(sorted(NO_RECHARGE))
    torch = pytest.importorskip("torch", reason="twin needs torch")  # noqa: F841
    from varuna.build.twin import F_TABLE
    assert all(F_TABLE.get(c, 0.0) == 0.0 for c in NO_RECHARGE)
