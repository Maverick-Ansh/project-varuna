"""Build `data/rain_features.json`. The other half of the dataset that had no builder.

`build_stack.py` regenerates the rasters; this regenerates the per-storm rainfall vector the net
reads alongside them. Both were produced in a scratch cell, so a new city could not be added to
either. Together they make the csi_net dataset reproducible from the repo.

    python -m csi_net.build_rain --verify                     # check against the committed file
    python -m csi_net.build_rain --areas patna,bengaluru --out data/rain_features.json

Rainfall is sampled in UTC, at each area's twin centre (`twin_meta.pt`), not at a project-wide default:
sampling every city's rain at Patna's centre was a real bug in this codebase's history, and it
silently priced Patna's monsoon into every other city's calibration.

The source is Open-Meteo's archive pinned to ECMWF IFS 9 km, NOT ERA5 — see
`varuna.serve.weather.historical_rain_mm` for the measurement behind that pin. Every number this
file writes inherits that choice, so regenerating it under a different model would move every
rainfall input in the project by roughly 2.2x.

Note what this data cannot do, since the model is fed it as if it could: nine probe points
spanning a 15 km tile collapse to a single archive cell, so there is exactly one rainfall number
per tile per storm. RESULTS.md §5 measures what that is worth, which is nothing.
"""
import argparse
import datetime as _dt
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WINDOWS = (1, 2, 3, 5, 7, 14, 30)      # antecedent-total windows, days
PEAKS = (1, 3, 6, 24)                  # rolling-maximum windows, hours
PEAK_WINDOW_D = 7                      # ...taken over the 7 days ending on the overpass
TZ = "UTC"                             # the day boundary these features use. NOT the twin's.


def rolling_max(series, k):
    """Largest sum of any k consecutive hours."""
    if len(series) < k:
        return round(float(sum(series)), 1)
    run = sum(series[:k])
    best = run
    for i in range(k, len(series)):
        run += series[i] - series[i - k]
        best = max(best, run)
    return round(float(best), 1)


def features_for(lat, lon, date):
    """The 12 numbers the committed file carries for one storm.

    `rain_Nd` is N calendar days ENDING ON `date` inclusive, so rain_1d is the overpass day
    itself. Open-Meteo's archive range is inclusive at both ends, which makes `d - N` a very
    easy off-by-one: it fetches N+1 days. `varuna/build/calibrate.py:_rain_for_date` has that
    off-by-one and also uses IST, so the twin's "2-day antecedent rain" is three IST days while
    csi_net's rain_2d is two UTC days. Both are defensible; they are not the same number, and
    until now nothing said so.
    """
    from varuna.serve.weather import historical_rain_mm, hourly_rain_series
    d1 = _dt.date.fromisoformat(date)
    out = {}
    for w in WINDOWS:
        d0 = d1 - _dt.timedelta(days=w - 1)
        out[f"rain_{w}d"] = round(historical_rain_mm(lat, lon, d0.isoformat(), d1.isoformat(),
                                                     timezone=TZ), 1)
    # Peaks are the heaviest burst anywhere in the 7 days ending on the overpass, not inside
    # the overpass day. Recovered by search, not from documentation: 3d and 5d windows fit some
    # dates, only 7d fits 117 of 120 committed peak values across all three areas.
    _, hourly = hourly_rain_series(
        lat, lon, (d1 - _dt.timedelta(days=PEAK_WINDOW_D - 1)).isoformat(), d1.isoformat(),
        timezone=TZ)
    for h in PEAKS:
        out[f"peak_{h}h"] = rolling_max(hourly, h)
    out["doy"] = d1.timetuple().tm_yday
    return out


def dates_for(area, artifacts):
    import glob
    return sorted(os.path.basename(f)[len("observed_water_"):-4] for f in
                  glob.glob(os.path.join(artifacts, area, "observed_water_*.tif")))


def centre_of(area, artifacts):
    from varuna.build.twin import _bundle_meta
    c = _bundle_meta(os.path.join(artifacts, area)).get("center")
    if not c:
        from varuna.areas import get_area
        c = get_area(area).center
    return [float(c[0]), float(c[1])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default="patna,mumbai_harbour,mumbai_northeast")
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--out", default="")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    ref_path = os.path.join(HERE, "data", "rain_features.json")
    ref = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
    out, diffs = {}, []
    for area in [a for a in args.areas.split(",") if a]:
        lat, lon = centre_of(area, args.artifacts)
        dates = dates_for(area, args.artifacts)
        print(f"{area:<18} centre ({lat:.4f}, {lon:.4f})  {len(dates)} date(s)")
        rain = {}
        for d in dates:
            rain[d] = features_for(lat, lon, d)
            if args.verify and area in ref and d in ref[area]["rain"]:
                for k, v in rain[d].items():
                    r = ref[area]["rain"][d].get(k)
                    if r is not None and abs(v - r) > 0.15:
                        diffs.append((area, d, k, r, v))
        out[area] = {"center": [lat, lon], "rain": rain}

    if args.verify:
        n = sum(len(v["rain"]) * (len(WINDOWS) + len(PEAKS) + 1) for v in out.values())
        print(f"\n{len(diffs)} of {n} values differ from the committed file by > 0.15 mm")
        for a, d, k, r, v in diffs[:25]:
            print(f"   {a:<18}{d}  {k:<10} committed {r:>8.1f}  rebuilt {v:>8.1f}")
        if len(diffs) > 25:
            print(f"   ... and {len(diffs) - 25} more")
        return out

    dest = args.out or ref_path
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)
    return out


if __name__ == "__main__":
    main()
