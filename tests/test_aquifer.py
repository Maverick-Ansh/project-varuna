"""V3 aquifer: infiltration metered by per-cell soil storage capacity.

The test that proves the aquifer is real: a cell at capacity must stop infiltrating.
Synthetic torch domains; build_domain checks ride the synth bundle. No network, no EE.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from varuna.build.twin import Domain  # noqa: E402


def _domain(n=32, infil_mm_hr=20.0, capacity=None, seed=0):
    g = np.random.default_rng(seed)
    z = 50.0 + 0.001 * np.arange(n)[None, :] + g.normal(0, 0.02, (n, n))
    infil = np.full((n, n), infil_mm_hr) / 1000.0 / 3600.0
    dom = Domain(z, np.full((n, n), 0.04), infil, np.zeros((n, n), dtype="float32"),
                 dx=60.0, device="cpu", capacity=capacity)
    dom.wc = np.full((n, n), 10, dtype=np.int32)
    return dom


def test_cell_at_capacity_stops_infiltrating():
    """60 mm of rain against a 5 mm soil column: every cell absorbs exactly its capacity and
    not a drop more; the rest stays on the surface. The closure stays exact."""
    cap = 0.005
    dom = _domain(capacity=np.full((32, 32), cap))
    out = dom.rollout(dom.z0, rain_mm=60.0, storm_hr=0.5, total_hr=1.0, track_infil=True)
    grid = out["infil_grid"].cpu().numpy()
    assert grid.max() <= cap + 1e-9
    assert np.isclose(grid, cap).mean() > 0.95          # ample rain -> soil fills everywhere
    cell = dom.dx * dom.dx
    rain_m3 = 60.0 / 1000.0 * 32 * 32 * cell
    assert abs(rain_m3 - float(out["volume"][-1]) - grid.sum() * cell) < 0.005 * rain_m3


def test_none_capacity_matches_huge_capacity_exactly():
    """capacity=None is the legacy rate-only model; a huge finite capacity must be identical
    (the clamp only ever binds when soil can actually fill)."""
    a = _domain(capacity=None)
    b = _domain(capacity=np.full((32, 32), 1e6))
    ha = a.simulate(a.z0, rain_mm=80.0, storm_hr=0.5, total_hr=1.0)
    hb = b.simulate(b.z0, rain_mm=80.0, storm_hr=0.5, total_hr=1.0)
    assert torch.equal(ha, hb)


def test_capacity_increases_predicted_flooding():
    """The honesty note made testable: adding soil storage makes the twin LESS able to absorb
    water, so final ponding must rise. If a physics change flattered the intervention, this
    would catch the sign."""
    free = _domain(capacity=None)
    metered = _domain(capacity=np.full((32, 32), 0.005))
    vf = free.rollout(free.z0, 60.0, storm_hr=0.5, total_hr=1.0)["volume"][-1]
    vm = metered.rollout(metered.z0, 60.0, storm_hr=0.5, total_hr=1.0)["volume"][-1]
    assert vm > vf


def test_gradient_flows_through_capacity_clamp():
    """calibrate.py multiplies dom.infil and backprops through simulate(grad=True); the
    capacity min() must pass a finite gradient, not detach the graph."""
    dom = _domain(n=16, capacity=np.full((16, 16), 0.02))
    dom.infil = dom.infil.clone().requires_grad_(True)
    hmax = dom.simulate(dom.z0, rain_mm=40.0, storm_hr=0.25, total_hr=0.5, grad=True)
    hmax.sum().backward()
    g = dom.infil.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_build_domain_capacity_and_ksat(synth_bundle):
    """synth bundle (sand 40% / clay 30%): capacity = Cosby porosity x root zone x avail_frac,
    and the infiltration rate is min(F_TABLE x soil_factor, cosby_ksat)."""
    from varuna.build.recharge import cosby_ksat
    from varuna.build.twin import build_domain
    from varuna.config import CFG

    dom = build_domain(synth_bundle, device="cpu")
    porosity = 0.505 - 0.00142 * 40.0 - 0.00037 * 30.0
    expect = porosity * CFG.root_zone_m * CFG.soil_avail_frac
    cap = dom.capacity.cpu().numpy()
    assert np.allclose(cap, expect, atol=1e-4)
    # rates: tree 10 * 0.6 (clay 30%) = 6 mm/hr, built 1 * 0.6 = 0.6 — both under ksat ~13
    infil_mm = dom.infil.cpu().numpy() * 1000.0 * 3600.0
    ksat = float(cosby_ksat(40.0, 30.0))
    assert infil_mm.max() <= ksat + 1e-6
    wc = dom.wc
    assert np.allclose(infil_mm[wc == 10], 6.0, atol=0.01)
    assert np.allclose(infil_mm[wc == 50], 0.6, atol=0.01)


def test_real_shallow_water_table_shrinks_capacity(synth_bundle):
    """A REAL nearby station with a 0.4 m water table must cap the soil column; the fake
    SAMPLE wells must not (Phase 0 gate feeds the physics)."""
    import pandas as pd
    from varuna.build.twin import build_domain
    from varuna.config import CFG

    dom_rootzone = build_domain(synth_bundle, device="cpu")
    pd.DataFrame([("Ghat_PZ1", 25.625, 85.084, 0.4)],
                 columns=["station", "lat", "lon", "depth_to_water_m"]
                 ).to_csv(f"{synth_bundle}/gw_levels.csv", index=False)
    dom_gw = build_domain(synth_bundle, device="cpu")
    ratio = float(dom_gw.capacity.mean() / dom_rootzone.capacity.mean())
    assert ratio == pytest.approx(0.4 / CFG.root_zone_m, abs=1e-3)


def test_budget_reports_runoff_and_capacity_terms():
    from varuna.serve.waterbalance import _budget_for_rain
    dom = _domain(capacity=np.full((32, 32), 0.01))
    e = _budget_for_rain(dom, 60.0, storm_hr=0.5, total_hr=1.0)
    assert e["runoff_out_m3"] == 0
    assert e["closure_err_pct"] < 0.5
    assert e["soil_capacity_m3"] > 0
    assert 0 < e["soil_filled_pct"] <= 100.0
