"""Groundwater-provenance tests: the SAMPLE warning must actually fire, and out-of-AOI
stations must be rejected (the bug where Patna's placeholder wells ranked Bengaluru).
No network, no Earth Engine; rasterio only via synth_bundle."""
import numpy as np
import pandas as pd
import pytest

from varuna.build import recharge

REAL_PATNA_STATIONS = pd.DataFrame(
    [("Danapur_PZ1", 25.633, 85.046, 8.5),
     ("PatnaSadar_PZ2", 25.610, 85.140, 6.0),
     ("Fatuha_PZ3", 25.560, 85.290, 10.3)],
    columns=["station", "lat", "lon", "depth_to_water_m"])

BENGALURU = (12.940, 77.640)
PATNA = (25.605, 85.140)


def test_sample_file_written_and_flagged(tmp_path):
    gw = recharge.load_groundwater(str(tmp_path))
    assert (tmp_path / "gw_levels.csv").exists()
    assert gw["_is_sample"].all()


def test_out_of_aoi_stations_rejected(tmp_path):
    """Real-named Patna stations must not pass for a Bengaluru-centred area (~1,400 km away)."""
    REAL_PATNA_STATIONS.to_csv(tmp_path / "gw_levels.csv", index=False)
    gw = recharge.load_groundwater(str(tmp_path), center=BENGALURU)
    assert gw["_is_sample"].all()
    assert gw["_km_to_center"].min() > 1000


def test_nearby_real_stations_accepted(tmp_path):
    REAL_PATNA_STATIONS.to_csv(tmp_path / "gw_levels.csv", index=False)
    gw = recharge.load_groundwater(str(tmp_path), center=PATNA)
    assert not gw["_is_sample"].any()


def test_station_km_haversine():
    """Sanity-pin the distance helper: Patna->Bengaluru is ~1,600 km; a point to itself is 0."""
    d = recharge.station_km(PATNA, [BENGALURU[0]], [BENGALURU[1]])
    assert 1500 < float(d[0]) < 1700
    assert float(recharge.station_km(PATNA, [PATNA[0]], [PATNA[1]])[0]) < 1e-6


def test_groundwater_status_reports_sample_and_reason(tmp_path):
    """Without depth.tif the AOI gate can't run — that must be reported, not silently passed."""
    status = recharge.groundwater_status(str(tmp_path))
    assert status["sample"] is True
    assert "SAMPLE" in status["reason"]
    assert status["aoi_check"].startswith("skipped")
    assert "does not certify" in status["caveat"]


def test_groundwater_status_out_of_aoi(tmp_path):
    REAL_PATNA_STATIONS.to_csv(tmp_path / "gw_levels.csv", index=False)
    status = recharge.groundwater_status(str(tmp_path), center=BENGALURU)
    assert status["sample"] is True
    assert "within" in status["reason"]
    assert status["nearest_station_km"] > 1000


def test_bundle_center_from_raster(synth_bundle):
    lat, lon = recharge.bundle_center(synth_bundle)
    # synth raster: 256 px at 0.00027 deg from top-left (85.05, 25.66)
    assert np.isclose(lat, 25.66 - 128 * 0.00027, atol=1e-6)
    assert np.isclose(lon, 85.05 + 128 * 0.00027, atol=1e-6)


def _write_sites(work):
    pd.DataFrame([dict(sink_id=1, lat=25.63, lon=85.08, volume_m3=5000.0,
                       rsi=0.8, recharge_score=4000.0)]).to_csv(f"{work}/recharge_sites.csv",
                                                                index=False)


def test_tools_recharge_warning_fires_on_sample_bundle(synth_bundle):
    """The pinned dead-code bug: the old check read a 'station' column off recharge_sites.csv
    (which has none), so the warning never fired even on 100% fake groundwater."""
    from varuna.agent import tools
    _write_sites(synth_bundle)
    out = tools.dispatch("recharge_sites", {"top_n": 5})
    assert out["sample"] is True
    assert out["warning"] is not None and "SAMPLE" in out["warning"]
    assert "does not certify" in out["caveat"]
    assert len(out["sites"]) == 1


def test_tools_recharge_no_warning_on_real_nearby_stations(synth_bundle):
    from varuna.agent import tools
    _write_sites(synth_bundle)
    # real-named stations right at the synth raster centre -> both checks pass
    pd.DataFrame([("Ghat_PZ1", 25.625, 85.084, 7.0), ("Ghat_PZ2", 25.630, 85.090, 6.5)],
                 columns=["station", "lat", "lon", "depth_to_water_m"]
                 ).to_csv(f"{synth_bundle}/gw_levels.csv", index=False)
    out = tools.dispatch("recharge_sites", {"top_n": 5})
    assert out["sample"] is False
    assert out["warning"] is None
