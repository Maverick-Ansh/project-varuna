"""Offline tests for metered recharge siting (no GEE needed; one test uses rasterio).

The claims under test are the plan's honesty properties, not magnitudes:
- the reported m³ comes from RE-SIMULATION (rollout call-count), not volume × score;
- the Phase-2 aquifer meters it — a full soil column yields ~zero added recharge no matter
  how many structures are placed;
- sites land ONLY on pervious ground, away from buildings (buffer) and off roads;
- solver blowups are flagged unstable, mirroring containers' guard.
"""
import numpy as np
import pytest
import torch

from varuna.build.landcover import PERVIOUS
from varuna.build.twin import Domain
from varuna.serve import recharge_sites as RS


def _synth_domain(N=40, capacity=None, infil=1e-6):
    """Torch-only pervious-crop domain with a bowl in the middle and a built block in a corner."""
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (0.02 * xx + 0.01 * yy).astype("float32")
    z -= 2.0 * np.exp(-(((xx - N * 0.5) ** 2 + (yy - N * 0.5) ** 2) / (2 * (N / 8.0) ** 2)))
    mann = np.full((N, N), 0.04, "float32")
    inf = np.full((N, N), infil, "float32")
    built = np.zeros((N, N), "float32")
    built[:6, :6] = 1.0
    cap = None if capacity is None else np.full((N, N), capacity, "float32")
    dom = Domain(z.astype("float32"), mann, inf, built, dx=60.0, device="cpu", capacity=cap)
    wc = np.full((N, N), 40, np.int32)          # crop — pervious
    wc[:6, :6] = 50                             # a built corner
    wc[N - 5:, N - 5:] = 80                     # a water corner
    dom.wc = wc
    return dom


def test_full_aquifer_yields_no_recharge(tmp_path):
    """The anti-flattery property: when the soil column is already full, structures add ~nothing."""
    # 5 mm of room, 18 mm/hr intake, 120 mm storm -> baseline saturates the aquifer everywhere
    dom = _synth_domain(capacity=0.005, infil=5e-6)
    rep = RS.plan_recharge(rain_mm=120.0, work=str(tmp_path), site_counts=(5, 20), dom=dom)

    assert rep["metered"] is True
    cell = 60.0 * 60.0
    cap_m3 = rep["soil_capacity_m3"]
    assert rep["base_infil_m3"] <= cap_m3 + 1            # metering holds at baseline
    assert rep["base_infil_m3"] >= 0.9 * cap_m3          # and it actually bound (column filled)
    for c in rep["curve"]:
        if not c["unstable"]:
            assert abs(c["recharge_m3"]) <= 0.02 * cap_m3 + cell  # nothing left to add
    # provenance still ships: synthetic work dir has only the auto-written SAMPLE gw file
    assert rep["gw_status"]["sample"] is True
    assert (tmp_path / "recharge_plan.json").exists()


def test_sites_pervious_buffered_and_off_roads(tmp_path):
    dom = _synth_domain()
    N = dom.z0.shape[0]
    # buildings right at the bowl's edge + a road stripe through the bowl
    b = np.zeros((N, N), "float32")
    b[N // 2 - 6:N // 2 - 3, N // 2 - 2:N // 2 + 2] = 1.0
    r = np.zeros((N, N), "float32")
    r[N // 2 + 3, :] = 1.0
    np.savez(tmp_path / "urban_grid.npz", buildings=b, roads=r)

    rep = RS.plan_recharge(rain_mm=120.0, work=str(tmp_path), site_counts=(5, 20), dom=dom,
                           building_buffer_cells=2)
    assert rep["sites"], "explicit site list must be emitted"
    from scipy import ndimage
    forbidden = ndimage.binary_dilation(b > 0, structure=np.ones((3, 3), bool), iterations=2)
    for s in rep["sites"]:
        rr, cc = s["row"], s["col"]
        assert int(dom.wc[rr, cc]) in PERVIOUS, "recharge structure must sit on pervious ground"
        assert not forbidden[rr, cc], "must respect the building safety buffer"
        assert r[rr, cc] == 0, "no recharge structure in a road"
    assert [s["rank"] for s in rep["sites"]] == list(range(1, len(rep["sites"]) + 1))


def test_measured_by_resimulation_and_ksat_boost(synth_bundle):
    """Integration on the rasterized bundle: rollout is actually called once per curve point
    plus the baseline, sites carry Ksat + lat/lon, and the boosted structures put water in."""
    from varuna.build.twin import build_domain
    # centre the crop on the conftest bowl (raster row 64 / col 80), small grid for speed
    dom = build_domain(synth_bundle, center=(25.6427, 85.0716), n_grid=48, device="cpu")
    assert dom.capacity is not None                     # sand/clay in bundle -> aquifer on

    calls = {"n": 0}
    orig = dom.rollout
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    dom.rollout = counting

    rep = RS.plan_recharge(rain_mm=120.0, work=synth_bundle, site_counts=(5, 15), dom=dom)
    n_points = len(rep["curve"])
    assert calls["n"] == 1 + n_points, "every curve point must be a fresh re-simulation"

    stable = [c for c in rep["curve"] if not c["unstable"]]
    assert stable, "synthetic bowl must yield stable points"
    assert max(c["recharge_m3"] for c in stable) > 0    # Ksat-opened basins put water underground
    assert rep["metered"] and rep["base_infil_m3"] <= rep["soil_capacity_m3"] + 1
    for s in rep["sites"][:10]:
        assert "ksat_mm_hr" in s and s["ksat_mm_hr"] > 0
        assert "latlon" in s and 25.5 < s["latlon"][0] < 25.7
    # schema mirror: the storage-panel keys exist
    for key in ("curve", "targets", "sites", "phases", "max_sites", "unit_m3"):
        assert key in rep
    ph = [p for p in rep["phases"] if p.get("reachable")]
    for p in ph:
        # cost is computed from the un-rounded volume; allow the ±0.5 m³ rounding of add_storage_m3
        assert abs(p["add_cost_inr"] - p["add_storage_m3"] * rep["recharge_inr_per_m3"]) \
            <= rep["recharge_inr_per_m3"]


def test_unstable_flagging():
    curve = RS._flag_unstable([
        dict(sites=5, recharge_m3=100.0, reduction_pct=float("nan"), flood_cut_pct=0.0),
        dict(sites=10, recharge_m3=-9e9, reduction_pct=-80.0, flood_cut_pct=0.0),
        dict(sites=20, recharge_m3=50.0, reduction_pct=3.0, flood_cut_pct=-80.0),
        dict(sites=40, recharge_m3=80.0, reduction_pct=5.0, flood_cut_pct=1.0),
    ])
    assert [c["unstable"] for c in curve] == [True, True, True, False]


def test_plot_recharge_dose(tmp_path):
    pytest.importorskip("matplotlib", reason="matplotlib not installed")
    dom = _synth_domain(N=28)
    rep = RS.plan_recharge(rain_mm=100.0, work=str(tmp_path), site_counts=(5, 15), dom=dom)
    out = RS.plot_recharge_dose(rep, out=str(tmp_path / "recharge_dose.png"))
    assert (tmp_path / "recharge_dose.png").exists() and out.endswith("recharge_dose.png")
