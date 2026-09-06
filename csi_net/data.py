"""Loading, pooling and splitting the leakage-free flood-extent dataset.

The stack ships static terrain at the native 30 m grid. Pooling to the 60 m scoring grid
happens here, so the same file serves both the 60 m run (comparable with the existing
twin / TWI / HAND numbers) and the 30 m resolution experiment.

SAR truth is max-pooled and features are mean-pooled, mirroring varuna.build.calibrate
exactly - if we pooled the truth differently the numbers would not be comparable.
"""
import datetime as _dt
import json

import numpy as np

RAIN_KEYS = ["rain_1d", "rain_3d", "rain_7d", "rain_30d", "peak_1h", "peak_3h", "peak_24h"]
N_RAIN = len(RAIN_KEYS) + 2                       # + doy_sin, doy_cos
N_TIDE = 4                                        # sin/cos of the synodic phase and its 2nd harmonic

SYNODIC = 29.530588853                            # mean lunar month, days
NEW_MOON = _dt.date(2000, 1, 6)                   # reference new moon (2000-01-06 18:14 UTC)


def tide_vec(date):
    """A tide proxy derived from the calendar alone - no tide gauge, no FES2014, no download.

    Sentinel-1 crosses at a fixed local solar time, so what varies between overpasses is the
    tide's phase at that hour. The lunar semidiurnal constituent drifts ~50 min/day against the
    clock, which aliases to a ~14.8-day cycle at a fixed observation time, and the M2/S2 beat
    (spring-neap) runs at the 29.5-day synodic month. Both are functions of the date, so the
    sin/cos of the synodic phase and of its second harmonic span the two periods that matter.

    This is a proxy, not a tide model: it carries phase, not amplitude, and knows nothing about
    local bathymetry or surge. It is enough to answer one question - whether a tidally forced
    domain becomes learnable once the model is told where in the tidal cycle each scene sits.
    """
    d = _dt.date.fromisoformat(str(date))
    ph = ((d - NEW_MOON).days % SYNODIC) / SYNODIC * 2 * np.pi
    return np.array([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph)], dtype="float32")

# Real flood events, identified by observed wet count running far above the domain's baseline.
FLOOD_DATES = {("patna", "2025-08-02"), ("mumbai_northeast", "2026-07-08")}
# Patna 2024-07-07 has zero observed wet cells: no ground truth, scores a forced 0.0 for
# every method and silently drags every mean down by 1/8.
DEAD_SCENES = {("patna", "2024-07-07")}

# Leave-one-DOMAIN-out is not leave-one-CITY-out once the dataset has more than one tile per
# city. Holding out mumbai_west still leaves five adjacent Mumbai tiles in training, and holding
# out Chitradurga leaves six other Karnataka towns sharing its climate, geology and tank
# morphology. The lodo number is therefore "unseen tile", which is a real result but not the
# deployment question. REGIONS groups domains that share a city or a region so `loro` can hold
# an entire one out: the model then meets a target with no sibling of any kind in training.
REGIONS = {
    "patna": "patna",
    "mumbai_south": "mumbai", "mumbai_west": "mumbai", "mumbai_east": "mumbai",
    "mumbai_north": "mumbai", "mumbai_northeast": "mumbai", "mumbai_harbour": "mumbai",
    "bengaluru": "karnataka", "doddaballapura": "karnataka", "ramanagara": "karnataka",
    "chitradurga": "karnataka", "chamarajanagara": "karnataka", "kolar": "karnataka",
    "chikkaballapur": "karnataka",
    # Chennai is its own region on purpose. It is 1,300 km from Mumbai and shares no catchment
    # with it, so holding one out leaves the other in training -- which is exactly the coastal
    # coverage test. Grouping the two as one "coastal" region would destroy that.
    "chennai": "chennai",
}


def region_of(area):
    return REGIONS.get(area, area)


def pool(a, k, agg):
    """Block-pool the trailing two axes by factor k."""
    if k == 1:
        return a
    *lead, H, W = a.shape
    b = a.reshape(*lead, H // k, k, W // k, k)
    return b.max(axis=(-3, -1)) if agg == "max" else b.mean(axis=(-3, -1))


class VarunaData:
    def __init__(self, npz_path, rain_json, grid_m=60, drop_dead=True, shuffle_rain=0,
                 keep_areas=None, tide=False):
        z = np.load(npz_path, allow_pickle=True)
        self.feature_names = [str(s) for s in z["feature_names"]]
        self.areas = [str(s) for s in z["areas"]]
        if keep_areas:
            # Mumbai-Harbour's rain<->extent correlation is significantly NEGATIVE
            # (r=-0.70, p=0.017 at 14 d): tidal, not pluvial. Training across it and
            # Mumbai-NE (r=+0.79) asks the net to fit two opposite stories at once.
            self.areas = [a for a in self.areas if a in keep_areas]
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
                v = self._rain_vec(rain, a, d)
                if tide:
                    v = np.concatenate([v, tide_vec(d)])
                self.samples.append(dict(area=a, date=d, idx=i, rain=v,
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
        seen = []
        for a in self.areas:
            if region_of(a) not in seen:
                seen.append(region_of(a))
        self.regions = seen
        self.tide = bool(tide)
        self.n_rain = N_RAIN + (N_TIDE if tide else 0)
        self.n_static = self.X[self.areas[0]].shape[0]
        self.n_in = self.n_static + self.n_rain

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
        lodo  - leave one domain out (one fold per area): spatial generalisation, unseen terrain.
                With several tiles per city this measures an unseen TILE, not an unseen city.
        loro  - leave one region out (one fold per city/region): the deployment question, with
                every sibling tile of the target held out alongside it.
        flood - train on quiet storms, test on the real flood events. Separates a model that
                predicts floods from one that has only memorised where water usually sits.
        """
        S = self.samples
        if kind == "loso":
            return [s for i, s in enumerate(S) if i != fold], [S[fold]]
        if kind == "lodo":
            a = self.areas[fold]
            return [s for s in S if s["area"] != a], [s for s in S if s["area"] == a]
        if kind == "loro":
            r = self.regions[fold]
            return ([s for s in S if region_of(s["area"]) != r],
                    [s for s in S if region_of(s["area"]) == r])
        if kind == "flood":
            return [s for s in S if not s["flood"]], [s for s in S if s["flood"]]
        raise ValueError(kind)

    def n_folds(self, kind):
        return {"loso": len(self.samples), "lodo": len(self.areas),
                "loro": len(self.regions), "flood": 1}[kind]

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
        rp = np.broadcast_to(r[:, None, None], (self.n_rain, H, W))
        return (np.concatenate([x, rp], 0).astype("float32"),
                self.Y[a][s["idx"]], self.valid[a])
