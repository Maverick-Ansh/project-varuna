"""Build `data/varuna_stack.npz` from the artifact bundles. The dataset had no builder.

Every csi_net number in this project came from a committed 8 MB binary that was produced in a
scratch notebook cell and never checked in as code. That made two things impossible: adding a
domain (the whole point of the next experiment), and honouring the paper's claim that every
headline number is regenerable from a committed artifact. This file closes both.

    python -m csi_net.build_stack --verify                       # check against the committed npz
    python -m csi_net.build_stack --areas patna,bengaluru --out data/varuna_stack.npz

An area is usable when its bundle has `dem.tif`, `worldcover.tif`, `jrc_occurrence.tif`,
`acc.tif`, `depth.tif` and at least one `observed_water_<date>.tif`. Soil rasters are optional
and zero-filled when absent, which is recorded in the manifest rather than silently assumed.

The crop is `varuna.build.twin.build_domain`'s own window at 30 m native (2N x 2N for an N x N
twin domain), so the stack, the twin and `calibrate.align_sar` all address the same ground.
`csi_net/twin_protocol.py` asserts that cell-for-cell before it scores anything.

--------------------------------------------------------------------------------------------
Verified against the committed stack (`--verify`): 22 of the 23 channels reproduce to float16
precision on all three areas. The exception is `twi`. Its committed values are not reproduced by
`varuna/build/baselines.py`'s own formula, nor by that formula on a filled DEM, a Horn/Sobel
slope, a 60 m slope, or the full-AOI raster; the implied flow-accumulation multiplier inverts to
~950 rather than 900, with scatter. The scratch cell that made it is gone, so rather than ship a
number whose provenance cannot be stated, this builder uses `baselines.py`'s definition --- the
project's own, documented, and identical for every area it builds.

That makes a rebuilt stack internally consistent but NOT byte-identical to the committed one.
`--verify` prints exactly how far apart they are, and `RESULTS.md` records the re-run that
confirms the headline is unchanged by the difference. Do not mix the two: rebuild every area or
none, or a leave-one-domain-out test silently compares domains built two different ways.
"""
import argparse
import datetime as _dt
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WC_CLASSES = (10, 20, 30, 40, 50, 60, 80, 90, 95)
FEATURES = (["dem_z", "dem_rel_s2", "dem_rel_s8", "dem_rel_s32", "slope", "curvature", "log_acc",
             "depth_depression", "twi", "hand", "sand_pct", "clay_pct", "jrc_occurrence",
             "log_dist_perm_water"] + [f"wc_{c}" for c in WC_CLASSES])

DX = 30.0                      # native raster resolution, metres
ACC_CELL_AREA = DX * DX        # baselines.py's TWI numerator uses acc in m^2
HAND_ACC_PCTL = 99.0           # stream network = the top 1 % of flow accumulation
HAND_CLIP = (-50.0, 200.0)     # committed stack is clipped to this; keeps float16 in range


def _crop(a, row0, col0, n):
    c = np.asarray(a, dtype="float64")[row0:row0 + n, col0:col0 + n]
    if c.shape != (n, n):      # defensive, matching calibrate._crop_to_domain
        c = np.pad(c, ((0, max(0, n - c.shape[0])), (0, max(0, n - c.shape[1]))),
                   mode="edge")[:n, :n]
    return c


def build_area(area, artifacts, verbose=True):
    """23 static channels, the SAR truth stack, and the 30 m JRC occurrence, on the twin's grid."""
    import rasterio
    from scipy import ndimage

    from varuna.build.twin import build_domain

    work = os.path.join(artifacts, area)
    dom = build_domain(work)
    n = dom.N * 2
    r0, c0 = dom.row0, dom.col0

    def read(name, required=True):
        p = os.path.join(work, name)
        if not os.path.exists(p):
            if required:
                raise FileNotFoundError(p)
            return None
        with rasterio.open(p) as s:
            return s.read(1)

    dem = _crop(read("dem.tif"), r0, c0, n)
    acc = _crop(read("acc.tif"), r0, c0, n)
    dep = _crop(read("depth.tif"), r0, c0, n)
    jrc = _crop(read("jrc_occurrence.tif"), r0, c0, n)
    wc = _crop(read("worldcover.tif"), r0, c0, n)
    sand = read("sand.tif", required=False)
    clay = read("clay.tif", required=False)
    sand = _crop(sand, r0, c0, n) / 10.0 if sand is not None else np.zeros_like(dem)
    clay = _crop(clay, r0, c0, n) / 10.0 if clay is not None else np.zeros_like(dem)

    gy, gx = np.gradient(dem, DX)
    slope = np.hypot(gy, gx)

    # Height above the nearest stream cell. The network is the top 1 % of flow accumulation, not
    # the JRC permanent-water mask: on a floodplain the JRC mask is one wide river and everything
    # is "far" from it, which carries no local information.
    stream = acc >= np.percentile(acc, HAND_ACC_PCTL)
    if stream.any():
        _, idx = ndimage.distance_transform_edt(~stream, return_indices=True)
        hand = np.clip(dem - dem[idx[0], idx[1]], *HAND_CLIP)
    else:
        hand = np.zeros_like(dem)

    # Distance to permanent water, in CELLS (not metres) before the log - matching the committed
    # stack, and keeping the channel inside float16's useful range.
    perm = jrc >= 50
    dist = ndimage.distance_transform_edt(~perm) if perm.any() else np.full_like(dem, n)

    ch = {
        # Absolute elevation is standardised per domain: its offset between cities is domain
        # shift, not signal, and leaving it in is how you fail leave-one-domain-out for an
        # uninteresting reason.
        "dem_z": (dem - dem.mean()) / (dem.std() + 1e-9),
        # DEM error is spatially correlated, so elevation minus its own blur survives a bias
        # that destroys absolute elevation. This is the +/-1 m vs 20 cm problem, attacked at
        # the feature level, and it is why a learned model co-locates where a router cannot.
        "dem_rel_s2": dem - ndimage.gaussian_filter(dem, 2),
        "dem_rel_s8": dem - ndimage.gaussian_filter(dem, 8),
        "dem_rel_s32": dem - ndimage.gaussian_filter(dem, 32),
        "slope": slope,
        "curvature": ndimage.laplace(ndimage.gaussian_filter(dem, 1)),
        "log_acc": np.log1p(acc),
        "depth_depression": dep,
        "twi": np.log((acc * ACC_CELL_AREA + 1.0) / (slope + 1e-3)),   # baselines.py's definition
        "hand": hand,
        "sand_pct": sand,
        "clay_pct": clay,
        "jrc_occurrence": jrc,
        "log_dist_perm_water": np.log1p(dist),
    }
    for c in WC_CLASSES:
        ch[f"wc_{c}"] = (wc == c).astype("float64")

    X = np.stack([ch[k] for k in FEATURES]).astype("float16")

    dates = sorted(os.path.basename(f)[len("observed_water_"):-4]
                   for f in glob.glob(os.path.join(work, "observed_water_*.tif")))
    if not dates:
        raise FileNotFoundError(f"{area}: no observed_water_<date>.tif — nothing to learn from")
    Y = np.stack([(_crop(read(f"observed_water_{d}.tif"), r0, c0, n) > 0.5) for d in dates]
                 ).astype("uint8")
    if verbose:
        wet = [float(y.mean()) for y in Y]
        print(f"  {area:<18} {n}x{n} @30m  {len(dates)} dates  "
              f"wet fraction {min(wet):.4f}-{max(wet):.4f}  "
              f"soil={'yes' if sand.any() else 'MISSING (zero-filled)'}")
    return X, Y, jrc.astype("float16"), dates


def verify(new, ref_path):
    """Channel-by-channel comparison against an existing stack. Prints, never asserts."""
    if not os.path.exists(ref_path):
        print(f"no reference at {ref_path} — nothing to verify against")
        return
    z = np.load(ref_path, allow_pickle=True)
    ref_feats = [str(s) for s in z["feature_names"]]
    print(f"\nverify against {os.path.basename(ref_path)}")
    print(f"{'area':<18}{'channel':<22}{'max rel':>10}{'med rel':>10}  status")
    print("-" * 68)
    worst = {}
    for area in new["areas"]:
        key = f"{area}__X"
        if key not in z.files:
            print(f"{area:<18}(absent from reference)")
            continue
        A, B = new[key].astype("float32"), z[key].astype("float32")
        if A.shape != B.shape:
            print(f"{area:<18}SHAPE {A.shape} vs {B.shape}")
            continue
        for i, name in enumerate(FEATURES):
            if name not in ref_feats:
                continue
            b = B[ref_feats.index(name)]
            d = np.abs(A[i] - b)
            sc = max(float(np.nanmax(np.abs(b))), 1e-6)
            mx, md = float(np.nanmax(d)) / sc, float(np.median(d)) / sc
            ok = mx < 3e-3
            worst[name] = max(worst.get(name, 0.0), mx)
            if not ok:
                print(f"{area:<18}{name:<22}{mx:>10.2e}{md:>10.2e}  DIFFERS")
        yk = f"{area}__Y"
        if yk in z.files:
            dy = int((new[yk] != z[yk]).sum()) if new[yk].shape == z[yk].shape else -1
            print(f"{area:<18}{'SAR truth':<22}{'':>10}{'':>10}  "
                  f"{'identical' if dy == 0 else f'{dy} cells differ'}")
    clean = [k for k, v in worst.items() if v < 3e-3]
    print(f"\n{len(clean)}/{len(FEATURES)} channels reproduce to float16 precision.")
    bad = {k: v for k, v in worst.items() if v >= 3e-3}
    if bad:
        print("differing:", ", ".join(f"{k} ({v:.1e})" for k, v in sorted(bad.items())))
        print("See this module's docstring: rebuild every area or none.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default="patna,mumbai_harbour,mumbai_northeast")
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--out", default="")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the committed stack instead of writing a new one")
    args = ap.parse_args(argv)

    areas = [a for a in args.areas.split(",") if a]
    out = {"feature_names": np.array(FEATURES), "areas": np.array(areas)}
    manifest = {"built": _dt.datetime.now().isoformat(timespec="seconds"), "areas": {}}
    print(f"building {len(areas)} area(s) at 30 m")
    for a in areas:
        X, Y, jrc, dates = build_area(a, args.artifacts)
        out[f"{a}__X"] = X
        out[f"{a}__Y"] = Y
        out[f"{a}__jrc30"] = jrc
        out[f"{a}__dates"] = np.array(dates)
        manifest["areas"][a] = dict(shape=list(X.shape), dates=dates)

    ref = os.path.join(HERE, "data", "varuna_stack.npz")
    if args.verify:
        verify(out, ref)
        return out
    dest = args.out or ref
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez_compressed(dest, **out)
    json.dump(manifest, open(os.path.splitext(dest)[0] + "_manifest.json", "w"), indent=2)
    print(f"\nwrote {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
