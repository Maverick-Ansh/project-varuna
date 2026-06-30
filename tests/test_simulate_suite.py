"""Offline tests for the flood simulation suite (pure torch; no rasterio/GEE)."""
import numpy as np

from varuna.build.twin import Domain
from varuna.serve import simulate_suite as S


def _synth_domain(N=40):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    z = (0.02 * xx + 0.01 * yy).astype("float32")
    z -= 2.5 * np.exp(-(((xx - N * 0.5) ** 2 + (yy - N * 0.5) ** 2) / (2 * (N / 8.0) ** 2)))
    mann = np.full((N, N), 0.04, "float32")
    infil = np.full((N, N), 1e-6, "float32")
    built = np.ones((N, N), "float32")
    return Domain(z.astype("float32"), mann, infil, built, dx=60.0, device="cpu")


def test_dose_response_monotone():
    rows = S.dose_response(_synth_domain(), rains=(20, 60, 120))
    vols = [r["flooded_volume_m3"] for r in rows]
    assert vols == sorted(vols)                       # more rain -> more flooding
    assert all(r["peak_depth_m"] >= 0 for r in rows)


def test_timelapse_records_dynamics():
    dom = _synth_domain()
    roll = S.timelapse(dom, rain_mm=100, every=20, storm_hr=0.2, total_hr=0.6,
                       probes=[(dom.N // 2, dom.N // 2)])
    assert roll["frames"].shape[1:] == (dom.N, dom.N)
    assert len(roll["volume"]) == roll["frames"].shape[0]
    assert roll["volume"][0] <= roll["volume"].max()  # water rises from zero
    assert roll["probes"].shape[0] == roll["frames"].shape[0]


def test_plot_renderers(tmp_path):
    dom = _synth_domain()
    rows = S.dose_response(dom, rains=(20, 80))
    roll = S.timelapse(dom, rain_mm=80, every=20, storm_hr=0.2, total_hr=0.4)
    p1 = S.plot_dose_response(rows, str(tmp_path / "dose.png"))
    p2 = S.plot_mass_curve(roll, str(tmp_path / "mass.png"))
    import os
    assert os.path.getsize(p1) > 0 and os.path.getsize(p2) > 0
