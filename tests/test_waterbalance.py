"""Water-balance: exact closure, class split, ladder interpolation, efficiency math."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _tiny_domain(n=48):
    from varuna.build.twin import Domain
    g = np.random.default_rng(0)
    z = 50 + 0.01 * np.arange(n)[None, :] + g.normal(0, 0.05, (n, n))
    wc = np.full((n, n), 10, dtype=np.int32)
    wc[10:20, 10:20] = 50
    wc[30:40, 5:15] = 30
    infil = np.where(wc == 50, 1.0, 8.0) / 1000.0 / 3600.0
    dom = Domain(z, np.full((n, n), 0.04), infil, (wc == 50).astype("float32"), dx=60)
    dom.wc = wc
    return dom


def test_closure_and_class_split():
    from varuna.serve.waterbalance import _budget_for_rain
    dom = _tiny_domain()
    e = _budget_for_rain(dom, 60.0, storm_hr=0.5, total_hr=1.0)
    assert e["closure_err_pct"] < 0.5
    assert abs(e["rain_m3"] - e["infiltrated_m3"] - e["ponded_final_m3"]) <= \
        0.005 * e["rain_m3"] + 2
    split = e["infil_by_class"]
    total_split = sum(split.values())
    assert abs(total_split - e["infiltrated_m3"]) <= max(2, 0.01 * e["infiltrated_m3"])
    # vegetation covers most cells and has the higher rate -> dominates
    assert split["vegetation"] > split["built"]


def test_counterfactual_ponds_more():
    from varuna.serve.waterbalance import _budget_for_rain, _counterfactual_ponded
    dom = _tiny_domain()
    e = _budget_for_rain(dom, 60.0, storm_hr=0.5, total_hr=1.0)
    cf = _counterfactual_ponded(dom, 60.0, storm_hr=0.5, total_hr=1.0)
    assert cf > e["ponded_final_m3"]          # concrete world floods more


def test_water_balance_interpolation(tmp_path):
    from varuna.serve.waterbalance import water_balance
    entries = [dict(rain_mm=r, rain_m3=r * 1000, infiltrated_m3=r * 600,
                    ponded_final_m3=r * 400, ponded_peak_m3=r * 500,
                    reduced_by_nature_m3=r * 200,
                    infil_by_class=dict(vegetation=r * 500, built=r * 50, bare=r * 50,
                                        water_wetland=0, other=0))
               for r in (10, 100, 250)]
    (tmp_path / "water_balance.json").write_text(json.dumps(dict(entries=entries, n_grid=64)))
    (tmp_path / "storage_sizing.json").write_text(json.dumps(dict(
        rain_mm=100, targets={"50%": dict(sites=10, storage_m3=100000, equiv_units=2000)})))
    out = water_balance(work=str(tmp_path), rain_mm=55.0, efficiency=0.5)
    assert out["rain_m3"] == 55000
    assert out["infiltrated_m3"] == 33000
    assert out["storable"]["50%"]["usable_m3"] == 50000     # 0.5 efficiency halves it
    # clamping outside the ladder
    hi = water_balance(work=str(tmp_path), rain_mm=999)
    assert hi["clamped_to"] == 250


def test_water_balance_missing_ladder(tmp_path):
    from varuna.serve.waterbalance import water_balance
    with pytest.raises(FileNotFoundError):
        water_balance(work=str(tmp_path), rain_mm=100)
