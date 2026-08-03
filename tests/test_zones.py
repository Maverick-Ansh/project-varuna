"""Named danger zones: clustering, ranking, naming and the honesty rules around all three."""
import numpy as np
import pytest

from varuna.serve.zones import (BANDS, band_for, danger_zones, haversine_km, name_for,
                                places_from_overpass)

pytest.importorskip("scipy", reason="scipy not installed")

N = 24


def _latlon(r, c):
    """A simple north-up mapper: row increases south, col increases east."""
    return [round(25.700 - 0.0006 * r, 6), round(85.100 + 0.0006 * c, 6)]


def _grid(blobs, n=N):
    """blobs: [(row, col, half, depth)] -> (hmax, built) with everything built-up."""
    h = np.zeros((n, n))
    for r, c, half, d in blobs:
        h[r - half:r + half + 1, c - half:c + half + 1] = d
    return h, np.ones((n, n))


def test_band_uses_the_citizen_report_vocabulary():
    assert band_for(1.4)[0] == "chest"
    assert band_for(0.9)[0] == "waist"
    assert band_for(0.5)[0] == "knee"
    assert band_for(0.2)[0] == "ankle"
    assert band_for(0.05)[0] == "damp"
    assert [b[0] for b in BANDS] == ["chest", "waist", "knee", "ankle"]   # deepest-first lookup


def test_severity_ladder_is_monotone():
    sevs = [band_for(d)[1] for d in (1.5, 0.9, 0.5, 0.2, 0.01)]
    assert sevs == ["EXTREME", "HIGH", "MODERATE", "LOW", "MINIMAL"]


def test_zones_are_ranked_by_built_flood_volume():
    """A shallow-but-wide pool outranks a deep-but-tiny one — volume is what a city cares about."""
    h, built = _grid([(5, 5, 1, 1.6), (16, 16, 4, 0.5)])
    z = danger_zones(h, built, _latlon, dx=60.0)
    assert len(z) == 2
    assert z[0]["volume_m3"] > z[1]["volume_m3"]
    assert z[0]["rank"] == 1 and z[1]["rank"] == 2
    assert z[0]["area_km2"] > z[1]["area_km2"]
    assert z[1]["band"] == "chest"                    # the small one is still the deepest


def test_dry_ground_makes_no_zone():
    h, built = _grid([])
    assert danger_zones(h, built, _latlon, dx=60.0) == []


def test_unbuilt_water_is_not_a_danger_zone():
    """The biggest wet blob on the Patna tile is the Ganga. A river is not a flood."""
    h, built = _grid([(6, 6, 3, 2.0), (17, 17, 2, 0.6)])
    built[:12, :] = 0.0                               # the deep blob sits on open water
    z = danger_zones(h, built, _latlon, dx=60.0)
    assert len(z) == 1
    assert z[0]["peak_depth_m"] == 0.6


def test_specks_are_ignored():
    h, built = _grid([(5, 5, 0, 1.0)])                # a single cell
    assert danger_zones(h, built, _latlon, dx=60.0, min_cells=2) == []
    assert len(danger_zones(h, built, _latlon, dx=60.0, min_cells=1)) == 1


def test_shallow_water_below_the_flood_threshold_is_ignored():
    h, built = _grid([(8, 8, 3, 0.10)])               # below CFG.min_depth_m (0.15)
    assert danger_zones(h, built, _latlon, dx=60.0) == []


def test_zone_is_named_after_the_nearest_place():
    h, built = _grid([(5, 5, 2, 1.0)])
    lat, lon = _latlon(5, 5)
    places = [{"name": "Rajendra Nagar", "lat": lat, "lon": lon, "kind": "suburb"},
              {"name": "Far Away", "lat": lat + 0.5, "lon": lon + 0.5, "kind": "town"}]
    z = danger_zones(h, built, _latlon, places=places, dx=60.0)
    assert z[0]["place"] == "Rajendra Nagar"
    assert z[0]["named"] is True
    assert z[0]["place_km"] < 0.2


def test_a_distant_name_is_hedged_not_claimed():
    """If the nearest name is kilometres off, the pin must say 'near X', not 'X'."""
    h, built = _grid([(5, 5, 2, 1.0)])
    lat, lon = _latlon(5, 5)
    places = [{"name": "Somewhere Else", "lat": lat + 0.09, "lon": lon, "kind": "town"}]
    z = danger_zones(h, built, _latlon, places=places, dx=60.0)
    assert z[0]["named"] is False
    assert z[0]["place"].startswith("near ")
    assert z[0]["place_km"] > 1.2


def test_zones_work_without_any_places_file():
    """A bundle that never fetched places still produces usable zones."""
    h, built = _grid([(5, 5, 2, 1.0)])
    z = danger_zones(h, built, _latlon, places=[], dx=60.0)
    assert z[0]["place"] == "unnamed area"
    assert z[0]["named"] is False
    assert z[0]["place_name"] is None


def test_pin_sits_on_the_water_not_the_bounding_box_centre():
    """An L-shaped pool must pin where the water actually is, not in the empty corner."""
    h = np.zeros((N, N))
    h[4:20, 4:7] = 0.3                                # vertical arm
    h[17:20, 4:20] = 0.3                              # horizontal arm
    h[18, 18] = 2.5                                   # the deep spot, far from the bbox centre
    built = np.ones((N, N))
    z = danger_zones(h, built, _latlon, dx=60.0)
    assert len(z) == 1
    pin_lat, pin_lon = z[0]["latlon"]
    deep_lat, deep_lon = _latlon(18, 18)
    assert haversine_km(pin_lat, pin_lon, deep_lat, deep_lon) < 0.6


def test_max_zones_caps_the_list():
    blobs = [(3 + 4 * i, 3 + 4 * j, 1, 0.5) for i in range(5) for j in range(5)]
    h, built = _grid(blobs, n=40)
    z = danger_zones(h, built, lambda r, c: [25.7 - 0.0006 * r, 85.1 + 0.0006 * c], dx=60.0,
                     max_zones=4)
    assert len(z) == 4
    assert [x["rank"] for x in z] == [1, 2, 3, 4]


def test_name_for_handles_an_empty_place_list():
    got = name_for(25.6, 85.1, [])
    assert got["label"] is None and got["name"] is None and got["named"] is False
    assert got["landmark"] is None


def test_a_neighbourhood_beats_a_slightly_closer_road():
    """People say 'Rajendra Nagar flooded', not 'the 400 m mark of Bailey Road flooded'."""
    lat, lon = 25.600, 85.140
    places = [
        {"name": "Bailey Road", "lat": lat + 0.0036, "lon": lon, "kind": "road"},        # ~400 m
        {"name": "Rajendra Nagar", "lat": lat + 0.0054, "lon": lon, "kind": "suburb"},   # ~600 m
    ]
    got = name_for(lat, lon, places)
    assert got["name"] == "Rajendra Nagar"
    assert got["landmark"] == "Bailey Road"        # the road still rides along as a landmark


def test_the_city_name_does_not_win_over_a_local_name():
    """'Patna' is true of every cell in the Patna domain, so it is useless as a flood label."""
    lat, lon = 25.600, 85.140
    places = [{"name": "Patna", "lat": lat, "lon": lon, "kind": "city"},
              {"name": "Kankarbagh", "lat": lat + 0.0045, "lon": lon, "kind": "suburb"}]
    assert name_for(lat, lon, places)["name"] == "Kankarbagh"


def test_a_road_names_the_zone_when_nothing_better_exists():
    """Patna has SEVEN place nodes and zero mapped wards — roads have to carry the naming."""
    lat, lon = 25.600, 85.140
    places = [{"name": "Ashok Rajpath", "lat": lat + 0.0009, "lon": lon, "kind": "road"}]
    got = name_for(lat, lon, places)
    assert got["name"] == "Ashok Rajpath"
    assert got["named"] is True
    assert got["landmark"] == "Ashok Rajpath"


def test_named_roads_become_a_run_of_anchors():
    """A 1 km road must be able to name a flood at EITHER end, not just at its midpoint."""
    geom = [{"lat": 25.600 + 0.0001 * i, "lon": 85.140} for i in range(100)]   # ~1.1 km
    osm = {"elements": [{"type": "way", "geometry": geom,
                         "tags": {"highway": "primary", "name": "Bailey Road"}}]}
    anchors = places_from_overpass(osm)["places"]
    assert len(anchors) >= 4, f"expected several anchors along the road, got {len(anchors)}"
    assert all(a["kind"] == "road" and a["name"] == "Bailey Road" for a in anchors)
    spread = max(a["lat"] for a in anchors) - min(a["lat"] for a in anchors)
    assert spread > 0.009                          # anchors span the whole road, not one point


def test_station_and_landuse_kinds_are_recognised():
    osm = {"elements": [
        {"type": "node", "lat": 25.60, "lon": 85.14,
         "tags": {"railway": "station", "name": "Rajendra Nagar Terminal"}},
        {"type": "way", "center": {"lat": 25.61, "lon": 85.15},
         "tags": {"landuse": "residential", "name": "Officers' Flat"}},
    ]}
    kinds = {p["name"]: p["kind"] for p in places_from_overpass(osm)["places"]}
    assert kinds["Rajendra Nagar Terminal"] == "station"
    assert kinds["Officers' Flat"] == "landuse"


def test_places_from_overpass_parses_nodes_and_boundary_centres():
    osm = {"elements": [
        {"type": "node", "lat": 25.6, "lon": 85.1, "tags": {"place": "suburb", "name": "Kankarbagh"}},
        {"type": "relation", "center": {"lat": 25.62, "lon": 85.12},
         "tags": {"boundary": "administrative", "admin_level": "9", "name": "Ward 22"}},
        {"type": "node", "lat": 25.63, "lon": 85.13, "tags": {"place": "suburb"}},   # unnamed
        {"type": "way", "tags": {"name": "No geometry"}},                            # no centre
    ]}
    got = places_from_overpass(osm, bbox=(25.5, 85.0, 25.7, 85.2))
    names = {p["name"] for p in got["places"]}
    assert names == {"Kankarbagh", "Ward 22"}
    assert got["places"][1]["kind"] == "admin9"
    assert got["bbox"] == [25.5, 85.0, 25.7, 85.2]


def test_places_from_overpass_dedupes_repeats():
    el = {"type": "node", "lat": 25.6, "lon": 85.1, "tags": {"place": "suburb", "name": "Dup"}}
    got = places_from_overpass({"elements": [el, dict(el)]})
    assert len(got["places"]) == 1
