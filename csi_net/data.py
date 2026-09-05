"""Loading, pooling and splitting the leakage-free flood-extent dataset.

The stack ships static terrain at the native 30 m grid. Pooling to the 60 m scoring grid
happens here, so the same file serves both the 60 m run (comparable with the existing
twin / TWI / HAND numbers) and the 30 m resolution experiment.

SAR truth is max-pooled and features are mean-pooled, mirroring varuna.build.calibrate
exactly - if we pooled the truth differently the numbers would not be comparable.
"""
import json
import numpy as np

RAIN_KEYS = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "peak_1h", "peak_3h", "peak_24h"]
N_RAIN = len(RAIN_KEYS) + 2                       # + doy_sin, doy_cos

# Real flood events, identified by observed wet count running far above the domain's baseline.
FLOOD_DATES = {("patna", "2025-08-02"), ("mumbai_northeast", "2026-07-08")}
# Patna 2024-07-07 has zero observed wet cells: no ground truth, scores a forced 0.0 for
# every method and silently drags every mean down by 1/8.
DEAD_SCENES = {("patna", "2024-07-07")}


def pool(a, k, agg):
    """Block-pool the trailing two axes by factor k."""
    if k == 1:
        return a
    *lead, H, W = a.shape
    b = a.reshape(*lead, H // k, k, W // k, k)
    return b.max(axis=(-3, -1)) if agg == "max" else b.mean(axis=(-3, -1))


class VarunaData:
    def __init__(self, npz_path, rain_json, grid_m=60, drop_dead=True, shuffle_rain=0):
        z = np.load(npz_path, allow_pickle=True)
        self.feature_names = [str(s) for s in z["feature_names"]]
        self.areas = [str(s) for s in z["areas"]]
        self.grid_m = grid_m
        k = {30: 1, 60: 2, 120: 4}[grid_m]
        rain = json.load(open(rain_json))

        self.X, self.Y, self.valid, self.dates, self.samples = {}, {}, {}, {}, []
        for a in self.areas:
            self.X[a] = pool(z[f"{a}__X"].astype("float32"), k, "mean")
            self.Y[a] = pool(z[f"{a}__Y"].astype("float32"), k, "max") > 0.5
            self.valid[a] = pool(z[f"{a}__jrc30"].astype("float32"), k, "mean") < 50
            self.dates[a] = [str(s) for s in z[f"{a}__dates"]]
            for i, d in enumerate(self.dates[a]):
                if drop_dead and (a, d) in DEAD_SCENES:
                    continue
                self.samples.append(dict(area=a, date=d, idx=i,
                                         rain=self._rain_vec(rain, a, d),
                                         flood=(a, d) in FLOOD_DATES))
        if shuffle_rain:
            # Permutation control: keep every rain vector, destroy only its pairing with the
            # storm it belongs to. Shuffled WITHIN an area, so each domain keeps its own rainfall
            # distribution and only the storm<->rain correspondence is broken. If the model scores
            # the same on shuffled rain as on real rain, the rain channels carry no information
            # about which storm this is - they are a scene fingerprint, not a forcing.
            rng = np.random.default_rng(shuffle_rain)
            for a in self.areas:
                idx = [i for i, s_ in enumerate(self.samples) if s_["area"] == a]
                perm = rng.permutation(len(idx))
                vecs = [self.samples[i]["rain"] for i in idx]
                for k, i in enumerate(idx):
                    self.samples[i]["rain"] = vecs[perm[k]]
        self.n_static = self.X[self.areas[0]].shape[0]
        self.n_in = self.n_static + N_RAIN

    @staticmethod
    def _rain_vec(rain, area, date):
        f = rain[area]["rain"][date]
        v = [np.log1p(f[k]) for k in RAIN_KEYS]                 # rain is heavy-tailed
        doy = f["doy"] * 2 * np.pi / 365.25
        return np.array(v + [np.sin(doy), np.cos(doy)], dtype="float32")

    # ------------------------------------------------------------------ splits
    def split(self, kind, fold):
        """Return (train_samples, test_samples).

        loso  - leave one storm out (30 folds): temporal generalisation, same terrain seen in training
        lodo  - leave one domain out (3 folds): spatial generalisation, unseen terrain. The hard test.
        flood - train on quiet storms, test on the real flood events. Separates a model that
                predicts floods from one that has only memorised where water usually sits.
        """
        S = self.samples
        if kind == "loso":
            return [s for i, s in enumerate(S) if i != fold], [S[fold]]
        if kind == "lodo":
            a = self.areas[fold]
            return [s for s in S if s["area"] != a], [s for s in S if s["area"] == a]
        if kind == "flood":
            return [s for s in S if not s["flood"]], [s for s in S if s["flood"]]
        raise ValueError(kind)

    def n_folds(self, kind):
        return {"loso": len(self.samples), "lodo": len(self.areas), "flood": 1}[kind]

    # ------------------------------------------------------------------ tensors
    def stats(self, samples, per_domain=True):
        """Channel mean/std for standardisation, computed on TRAIN samples only.

        per_domain re-centres each city separately, which removes the elevation/soil offset
        between Patna and Mumbai. That offset is domain shift, not signal, and leaving it in
        is the obvious way to fail the leave-one-domain-out test for an uninteresting reason.
        """
        out = {}
        if per_domain:
            for a in self.areas:
                x = self.X[a]
                out[a] = (x.mean((1, 2), keepdims=True), x.std((1, 2), keepdims=True) + 1e-6)
        else:
            areas = sorted({s["area"] for s in samples})
            cat = np.concatenate([self.X[a].reshape(self.n_static, -1) for a in areas], 1)
            m, sd = cat.mean(1)[:, None, None], cat.std(1)[:, None, None] + 1e-6
            for a in self.areas:
                out[a] = (m, sd)
        r = np.stack([s["rain"] for s in samples])
        out["_rain"] = (r.mean(0), r.std(0) + 1e-6)
        return out

    def tensor(self, s, stats):
        """Full-scene input (C,H,W), truth (H,W) and valid mask (H,W) for one storm."""
        a = s["area"]
        m, sd = stats[a]
        x = (self.X[a] - m) / sd
        rm, rsd = stats["_rain"]
        r = (s["rain"] - rm) / rsd
        H, W = x.shape[1:]
        rp = np.broadcast_to(r[:, None, None], (N_RAIN, H, W))
        return (np.concatenate([x, rp], 0).astype("float32"),
                self.Y[a][s["idx"]], self.valid[a])
