"""Score the twin against observed depths: right place, right day, real rain.

The procedure per observation:

  1. find which built area's domain contains the point (skip it if none does)
  2. ask Open-Meteo how much rain actually fell there in the `window_h` before the report
  3. run the emulator for that storm total
  4. read the predicted depth in the observation's own 60 m cell
  5. compare

Two things this deliberately does NOT do.

It does not tune anything. These observations are a held-out test set; the moment a threshold is
fitted to them they stop being one. (The nightly learner is fed citizen reports through a separate
path with its own reward gate — see `learn.reward`.)

It does not report a bare error. A depth model is easy to look good with: predict zero everywhere
and the mean error on a set where the median depth is 8 cm looks respectable. So every run scores
the same observations under three trivial baselines — always-dry, the observed mean, and the
observed median — and the model's number only means something relative to those. A model that
cannot beat "always predict the mean" has not learned where water goes.

The framing to keep honest: this asks "given the day's rainfall total, does the twin put the right
depth at this spot?" The emulator was trained on a 1.5 h design storm, not on the real hyetograph,
so a miss can be the storm shape as easily as the terrain.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math

import numpy as np

log = logging.getLogger("varuna.groundtruth.score")

WET_THRESHOLD_M = 0.15        # CFG.min_depth_m — "flooded" for the binary skill scores
DEFAULT_WINDOW_H = 24


def _areas_with_bounds(area_ids=None):
    """[(area_id, work, (south, west, north, east), center)] for built areas, from their bundles."""
    from ..areas import list_areas, is_built
    import rasterio
    import torch
    out = []
    for a in list_areas():
        if not is_built(a.id) or (area_ids and a.id not in area_ids):
            continue
        work = a.work_dir()
        try:
            meta = torch.load(f"{work}/twin_meta.pt", map_location="cpu", weights_only=False)
            with rasterio.open(f"{work}/dem.tif") as s:
                T = s.transform
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", a.id, e)
            continue
        r0, c0, N = meta["row0"], meta["col0"], meta["n_grid"]
        lon0, lat_top = T * (c0, r0)
        lon1, lat_bot = T * (c0 + N * 2, r0 + N * 2)
        bounds = (min(lat_top, lat_bot), min(lon0, lon1), max(lat_top, lat_bot), max(lon0, lon1))
        out.append((a.id, work, bounds, T, meta))
    return out


def assign_areas(observations, area_ids=None):
    """Tag each observation with the area whose domain contains it. Returns (tagged, unmatched)."""
    areas = _areas_with_bounds(area_ids)
    tagged, unmatched = [], []
    for o in observations:
        hit = None
        for aid, _work, (s, w, n, e), _T, _m in areas:
            if s <= o.lat <= n and w <= o.lon <= e:
                hit = aid
                break
        (tagged if hit else unmatched).append(o.with_area(hit) if hit else o)
    return tagged, unmatched


def antecedent_rain_mm(lat, lon, when, window_h=DEFAULT_WINDOW_H, cache=None):
    """Rain that actually fell in the `window_h` before `when` (a tz-aware datetime), in mm.

    Grouped by (~1 km cell, date) through `cache`: neighbouring reports on the same day share one
    upstream call, which keeps a 100-observation run to a couple of dozen requests.
    """
    from ..serve.weather import hourly_rain_series
    key = (round(lat, 2), round(lon, 2), when.date().isoformat())
    if cache is not None and key in cache:
        times, precip = cache[key]
    else:
        start = (when.date() - _dt.timedelta(days=2)).isoformat()
        end = when.date().isoformat()
        times, precip = hourly_rain_series(lat, lon, start, end)
        if cache is not None:
            cache[key] = (times, precip)
    if not times:
        return None
    # Open-Meteo returns local (Asia/Kolkata) naive stamps; compare in the same frame
    tz = when.tzinfo
    lo = when - _dt.timedelta(hours=window_h)
    total = 0.0
    seen = False
    for t, p in zip(times, precip):
        try:
            ts = _dt.datetime.fromisoformat(t)
        except ValueError:
            continue
        if ts.tzinfo is None and tz is not None:
            ts = ts.replace(tzinfo=tz)
        if lo <= ts <= when:
            total += float(p or 0)
            seen = True
    return round(total, 2) if seen else None


def _cell_of(T, meta, lat, lon, N):
    """(row, col) of a lat/lon inside the domain grid, or None if it falls outside."""
    inv = ~T
    col30, row30 = inv * (lon, lat)
    r = int((row30 - meta["row0"]) // 2)
    c = int((col30 - meta["col0"]) // 2)
    return (r, c) if (0 <= r < N and 0 <= c < N) else None


def predict_depths(observations, window_h=DEFAULT_WINDOW_H, area_ids=None, device="cpu",
                   rain_cache=None, progress=None, min_rain_mm=0.0):
    """Attach the real antecedent rainfall and the model's predicted depth to each observation.

    `min_rain_mm` drops observations that no plausible amount of rain could have caused: the live
    feed contains 1.42 m of standing water at Worli after 3.8 mm in 24 h. Whatever that is — a
    burst main, a high tide, a mistyped location, a test row — it is not rainfall-driven
    waterlogging, and a rainfall-driven model should be neither blamed nor credited for it.
    This is a physical screen on the INPUT, not a threshold tuned against the score; it is applied
    before any prediction is looked at, and the count of what it removed is reported.

    Returns (rows, skipped) where a row is a plain dict ready for `metrics` or a report.
    """
    from ..serve.emulator import whatif_grid
    areas = {aid: (work, T, meta) for aid, work, _b, T, meta in _areas_with_bounds(area_ids)}
    rain_cache = {} if rain_cache is None else rain_cache
    grid_cache = {}
    rows, skipped = [], []

    for i, o in enumerate(observations):
        if progress and i % 10 == 0:
            progress(i, len(observations))
        if not o.area or o.area not in areas:
            skipped.append((o, "no_area"))
            continue
        work, T, meta = areas[o.area]
        cell = _cell_of(T, meta, o.lat, o.lon, meta["n_grid"])
        if cell is None:
            skipped.append((o, "outside_grid"))
            continue
        try:
            rain = antecedent_rain_mm(o.lat, o.lon, o.dt(), window_h, cache=rain_cache)
        except Exception as e:  # noqa: BLE001
            log.warning("rain lookup failed for %s: %s", o.source_id, e)
            rain = None
        if rain is None:
            skipped.append((o, "no_rainfall_data"))
            continue
        if rain < min_rain_mm:
            skipped.append((o, "rain_too_low_to_be_pluvial"))
            continue
        gkey = (o.area, round(rain, 1))
        if gkey not in grid_cache:
            hmax, _dig, _s = whatif_grid(round(rain, 1), None, work=work, device=device)
            grid_cache[gkey] = hmax
        hmax = grid_cache[gkey]
        rows.append({
            "source": o.source, "source_id": o.source_id, "area": o.area,
            "lat": o.lat, "lon": o.lon, "ts": o.ts, "place": o.place,
            "observed_m": float(o.depth_m),
            "predicted_m": round(float(hmax[cell]), 3),
            "rain_mm": rain, "window_h": window_h,
            "cell": [int(cell[0]), int(cell[1])],
        })
    return rows, skipped


# --------------------------------------------------------------------------- metrics


def _errors(pred, obs):
    pred, obs = np.asarray(pred, dtype="float64"), np.asarray(obs, dtype="float64")
    d = pred - obs
    return {
        "mae_m": round(float(np.abs(d).mean()), 4),
        "rmse_m": round(float(np.sqrt((d ** 2).mean())), 4),
        "bias_m": round(float(d.mean()), 4),
        "median_abs_err_m": round(float(np.median(np.abs(d))), 4),
    }


def _binary(pred, obs, thresh=WET_THRESHOLD_M):
    """Hit/miss/false-alarm on 'was this spot flooded', plus CSI — comparable with the SAR work."""
    pred, obs = np.asarray(pred), np.asarray(obs)
    p, o = pred >= thresh, obs >= thresh
    tp, fp, fn = int((p & o).sum()), int((p & ~o).sum()), int((~p & o).sum())
    tn = int((~p & ~o).sum())
    denom = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "csi": round(tp / denom, 4) if denom else None,
            "pod": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "far": round(fp / (tp + fp), 4) if (tp + fp) else None,
            "threshold_m": thresh}


def metrics(rows, thresh=WET_THRESHOLD_M):
    """Model error, the trivial baselines it has to beat, and the skill score against the best."""
    if not rows:
        return {"n": 0}
    obs = [r["observed_m"] for r in rows]
    pred = [r["predicted_m"] for r in rows]
    mean_o, median_o = float(np.mean(obs)), float(np.median(obs))

    base = {
        "always_dry": _errors([0.0] * len(obs), obs),
        "observed_mean": _errors([mean_o] * len(obs), obs),
        "observed_median": _errors([median_o] * len(obs), obs),
    }
    model = _errors(pred, obs)
    best_base = min(base.values(), key=lambda b: b["rmse_m"])
    # Nash-Sutcliffe style: 1 = perfect, 0 = no better than the best trivial baseline, <0 = worse
    skill = 1.0 - (model["rmse_m"] ** 2) / (best_base["rmse_m"] ** 2) if best_base["rmse_m"] else None

    return {
        "n": len(rows),
        "observed": {"mean_m": round(mean_o, 4), "median_m": round(median_o, 4),
                     "max_m": round(float(np.max(obs)), 4),
                     "n_wet": int(np.sum(np.asarray(obs) >= thresh))},
        "predicted": {"mean_m": round(float(np.mean(pred)), 4),
                      "median_m": round(float(np.median(pred)), 4),
                      "max_m": round(float(np.max(pred)), 4),
                      "n_wet": int(np.sum(np.asarray(pred) >= thresh))},
        "model": model,
        "baselines": base,
        "skill_vs_best_baseline": None if skill is None else round(skill, 4),
        "binary": _binary(pred, obs, thresh),
        "correlation": _correlation(pred, obs),
    }


def _correlation(pred, obs):
    """Pearson r on the depths. Low-but-positive still means the model ranks places usefully even
    when the absolute depths are off; ~0 means it does not know which spot is worse."""
    p, o = np.asarray(pred, dtype="float64"), np.asarray(obs, dtype="float64")
    if len(p) < 3 or p.std() < 1e-12 or o.std() < 1e-12:
        return None
    return round(float(np.corrcoef(p, o)[0, 1]), 4)


def subgrid_dilution(rows, dx=60.0, wet=WET_THRESHOLD_M):
    """How much of the depth gap is just cell size? Returns the ratio and what it implies.

    A person reports the depth where they are standing — in the deepest part of the street. The
    model reports the MEAN depth of a 60 m x 60 m cell, 3,600 m2 of ground. Those are not the same
    quantity, and the ratio between them is a measurement, not an error: if the same volume of
    water sits in a pond of area A inside the cell, then

        observed / predicted  ~=  cell_area / A

    so the observed ratio implies the footprint the water actually occupies. On the Mumbai set
    this comes out at ~4.4x, implying ~800 m2 — a square 28 m across, which is a flooded street
    junction. That is a resolution artifact with a known cure (a finer grid), and it is worth
    separating from the ranking failure, which a finer grid would NOT cure.
    """
    pairs = [(r["observed_m"], r["predicted_m"]) for r in rows
             if r["observed_m"] >= wet and r["predicted_m"] > 1e-3]
    if len(pairs) < 5:
        return None
    ratio = float(np.median([o / p for o, p in pairs]))
    cell_area = dx * dx
    implied = cell_area / ratio if ratio > 0 else None
    return {
        "n": len(pairs),
        "median_observed_over_predicted": round(ratio, 3),
        "cell_area_m2": round(cell_area),
        "implied_ponding_area_m2": None if implied is None else round(implied),
        "implied_ponding_span_m": None if implied is None else round(math.sqrt(implied), 1),
        "note": ("Point observation vs cell mean. The ratio implies the sub-cell area the water "
                 "occupies; it is a resolution effect, not evidence the model has the volume wrong."),
    }


def neighbourhood_scores(rows, grid_fn, radii=(0, 1, 2, 3, 4), dx=60.0, thresh=WET_THRESHOLD_M):
    """Re-score allowing the model to be RIGHT NEARBY — standard neighbourhood verification.

    A 60 m model placing a flood one cell over is not really wrong, so meteorology scores against
    the best value in a box around the point rather than the single cell. Sweeping the box size
    separates two failure modes that look identical in a single number:

      * if error falls as the box grows, the model knows where the water is and is misplacing it
        slightly — a resolution/location problem
      * if error does NOT fall, the model does not know which place floods, and no amount of
        positional slack will save it

    `grid_fn(area, rain_mm) -> 2-D depth array` is injected so this stays testable.
    """
    obs = np.array([r["observed_m"] for r in rows], dtype="float64")
    wet = obs >= thresh
    out = {}
    for rad in radii:
        pred = []
        for r in rows:
            g = np.asarray(grid_fn(r["area"], r["rain_mm"]))
            rr, cc = r["cell"]
            r0, r1 = max(0, rr - rad), min(g.shape[0], rr + rad + 1)
            c0, c1 = max(0, cc - rad), min(g.shape[1], cc + rad + 1)
            pred.append(float(g[r0:r1, c0:c1].max()))
        p = np.array(pred)
        e = _errors(p, obs)
        out[str(rad)] = {
            "box_m": int((2 * rad + 1) * dx),
            **e,
            "predicted_mean_m": round(float(p.mean()), 4),
            "correlation": _correlation(p, obs),
            "correlation_wet_only": (_correlation(p[wet], obs[wet]) if wet.sum() >= 3 else None),
        }
    return out


def by_area(rows, thresh=WET_THRESHOLD_M):
    areas = sorted({r["area"] for r in rows})
    return {a: metrics([r for r in rows if r["area"] == a], thresh) for a in areas}


def report(rows, skipped=None, obs_set=None, thresh=WET_THRESHOLD_M):
    """The full validation payload: overall + per-area metrics, with the QC ledger attached."""
    out = {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "overall": metrics(rows, thresh),
        "by_area": by_area(rows, thresh),
        "n_scored": len(rows),
        "rain_source": "open-meteo (archive: ECMWF IFS 9 km / forecast past_days)",
        "note": ("Depth is compared where and when it was observed. The emulator is forced with "
                 "the observed rainfall TOTAL over the window and internally assumes a 1.5 h "
                 "design storm, so disagreement can come from storm shape as well as terrain. "
                 "Observations are scoring-only and never train the model."),
    }
    if skipped:
        counts = {}
        for _o, why in skipped:
            counts[why] = counts.get(why, 0) + 1
        out["skipped"] = counts
    if obs_set is not None:
        out["quality_control"] = {"rejected": obs_set.rejected, "kept": len(obs_set),
                                  **obs_set.meta}
    dil = subgrid_dilution(rows)
    if dil:
        out["subgrid_dilution"] = dil
    return out
