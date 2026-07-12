"""Citizen report store: validation, rate limits, persistence round-trip. No network."""
import pytest

from varuna.serve.reports import ReportStore, DEPTH_M, sanitize_note, ip_hash


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("varuna.serve.reports.cell_of", lambda w, la, lo: (10, 20))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return ReportStore(root=str(tmp_path))


def test_depth_bands():
    assert set(DEPTH_M) == {"ankle", "knee", "waist", "chest"}
    assert DEPTH_M["ankle"] < DEPTH_M["knee"] < DEPTH_M["waist"] < DEPTH_M["chest"]


def test_add_and_recent(store):
    rep = store.add("patna", "w", 25.6, 85.1, "knee", note="water rising", ip_hash="aa")
    assert rep["depth_m"] == 0.4 and rep["cell"] == [10, 20]
    assert "ip_hash" not in rep                       # public view is scrubbed
    rs = store.recent(area="patna")
    assert len(rs) == 1 and store.recent(area="other") == []
    assert store.summary(area="patna")["max_depth_m"] == 0.4


def test_bad_band_and_outside(store, monkeypatch):
    with pytest.raises(ValueError):
        store.add("patna", "w", 25.6, 85.1, "neck", ip_hash="aa")
    monkeypatch.setattr("varuna.serve.reports.cell_of", lambda w, la, lo: None)
    with pytest.raises(ValueError, match="outside_area"):
        store.add("patna", "w", 0.0, 0.0, "knee", ip_hash="aa")


def test_rate_limits(store, monkeypatch):
    cells = iter([(r, r) for r in range(100)])
    monkeypatch.setattr("varuna.serve.reports.cell_of", lambda w, la, lo: next(cells))
    for _ in range(5):
        store.add("patna", "w", 25.6, 85.1, "knee", ip_hash="same")
    with pytest.raises(ValueError, match="rate_limited"):
        store.add("patna", "w", 25.6, 85.1, "knee", ip_hash="same")
    store.add("patna", "w", 25.6, 85.1, "knee", ip_hash="other")   # different user OK


def test_same_cell_dedupe(store, monkeypatch):
    monkeypatch.setattr("varuna.serve.reports.cell_of", lambda w, la, lo: (5, 5))
    store.add("patna", "w", 25.6, 85.1, "knee", ip_hash="aa")
    with pytest.raises(ValueError, match="same spot"):
        store.add("patna", "w", 25.6, 85.1, "waist", ip_hash="aa")


def test_jsonl_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("varuna.serve.reports.cell_of", lambda w, la, lo: (1, 2))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s1 = ReportStore(root=str(tmp_path))
    for i in range(3):
        s1.add("patna", "w", 25.6, 85.1 + i * 1e-3, "ankle", ip_hash=f"h{i}")
    s2 = ReportStore(root=str(tmp_path))
    s2._hydrate()                                     # synchronous for the test
    assert len(s2.recent(area="patna")) == 3


def test_sanitize_and_hash():
    assert sanitize_note("a" * 500) and len(sanitize_note("a" * 500)) == 280
    assert sanitize_note("ok\x00\x07bad") == "okbad"
    assert ip_hash("1.2.3.4", salt="s") == ip_hash("1.2.3.4", salt="s")
    assert ip_hash("1.2.3.4", salt="s") != ip_hash("1.2.3.4", salt="t")
