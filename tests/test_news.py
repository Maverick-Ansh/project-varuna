"""News ingest, triage and QUARANTINE. No network (feed + LLM monkeypatched at the module
symbol), no torch/GEE needed. The adversarial tests are the point: a coordinated burst of
correlated coverage must never reach the learner, and a malformed LLM response must fall
back to the keyword template rather than raise."""
import datetime as _dt
from email.utils import format_datetime

import pytest


def _rss(*rows):
    items = "".join(
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>{format_datetime(_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=2))}"
        f"</pubDate><source url='https://x.in'>Outlet</source></item>"
        for title, link in rows)
    return (f"<?xml version='1.0'?><rss version='2.0'><channel>{items}"
            f"</channel></rss>").encode()


CANNED = _rss(
    ("Knee-deep water floods Silk Board after overnight downpour", "https://ex.com/a"),
    ("Civic budget approved for 2027", "https://ex.com/b"),
    ("Waterlogging shuts Outer Ring Road; 3 feet of water reported", "https://ex.com/c"),
)


# --------------------------------------------------------------------------- fetch + triage

def test_parse_and_keyword_triage(monkeypatch):
    import varuna.serve.news as N
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(N, "_fetch_url", lambda url, timeout=30: CANNED)
    items = N.fetch_news("bengaluru flood")
    assert len(items) == 3 and items[0]["outlet"] == "Outlet" and items[0]["published"]
    triaged, backend = N.classify(items)
    assert backend == "template"                       # no key -> deterministic path only
    assert triaged[0]["flood_related"] and triaged[0]["matched"] == "keywords"
    assert not triaged[1]["flood_related"]             # budget story is not a flood
    assert triaged[2]["flood_related"]


def test_provider_fallback_then_error(monkeypatch):
    import varuna.serve.news as N
    calls = []

    def flaky(url, timeout=30):
        calls.append(url)
        if "google" in url:
            raise RuntimeError("provider down")
        return CANNED

    monkeypatch.setattr(N, "_fetch_url", flaky)
    assert len(N.fetch_news("q")) == 3                 # second provider answered
    assert len(calls) == 2
    monkeypatch.setattr(N, "_fetch_url",
                        lambda url, timeout=30: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError, match="all news providers failed"):
        N.fetch_news("q")


def test_depth_snaps_to_exact_band_centre():
    """labels.py reverse-looks-up bands by float EQUALITY on DEPTH_M — every depth we emit
    must be an exact band centre, or a news depth would silently mislabel as 'knee'."""
    from varuna.serve.news import snap_depth
    from varuna.serve.reports import DEPTH_M
    for text, want in [("knee-deep water", "knee"), ("waist-deep on 100 Feet Road", "waist"),
                       ("water up to the chest", "chest"), ("3 feet of water", "waist"),
                       ("0.5 metres of water", "knee"), ("ankle deep slush", "ankle")]:
        band, depth = snap_depth(text)
        assert band == want
        # the exact reverse lookup the learner performs must find the band again
        assert next((b for b, m in DEPTH_M.items() if abs(m - depth) < 1e-6), None) == band
    assert snap_depth("roads closed for repairs") == (None, None)


# --------------------------------------------------------------------------- LLM path

def test_llm_refines_but_python_decides(monkeypatch):
    import varuna.serve.news as N
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(N, "_llm_call", lambda msgs, timeout=30.0:
                        '{"items": [{"i": 1, "flood": true, "location": "Old Town",'
                        ' "depth": "ankle-deep"}, {"i": 99, "flood": true}]}')
    triaged, backend = N.classify(N.parse_rss(CANNED))
    assert backend == "hosted"
    assert triaged[1]["flood_related"] and triaged[1]["matched"] == "llm"
    assert triaged[1]["depth_band"] == "ankle" and triaged[1]["depth_m"] == 0.1
    assert triaged[0]["flood_related"]                 # keyword hit survives the merge
    assert len(triaged) == 3                           # invented index 99 was ignored


def test_llm_malformed_falls_back(monkeypatch):
    import varuna.serve.news as N
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    for bad in (lambda m, timeout=30.0: "this is not json",
                lambda m, timeout=30.0: (_ for _ in ()).throw(RuntimeError("api down"))):
        monkeypatch.setattr(N, "_llm_call", bad)
        triaged, backend = N.classify(N.parse_rss(CANNED))
        assert backend == "template"                   # fell back, never raised
        assert triaged[0]["flood_related"]             # keyword triage intact


# --------------------------------------------------------------------------- store + payload

def test_store_provenance_dedupe_and_note(tmp_path, monkeypatch):
    import varuna.serve.news as N
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    store = N.NewsStore(root=str(tmp_path))
    triaged, backend = N.classify(N.parse_rss(CANNED))
    assert store.add_items("bengaluru", triaged, backend=backend) == 3
    assert store.add_items("bengaluru", triaged, backend=backend) == 0   # deduped on refetch
    store.mark_fetch("bengaluru", 3)
    feed = store.feed("bengaluru")
    assert all(r["source"] == "news" for r in feed["items"])             # provenance, always
    assert feed["trained_on"] is False and "NEVER used to train" in feed["note"]
    assert feed["coverage"] == "reported" and feed["count"] == 2         # flood-related only


def test_coverage_three_state(tmp_path, monkeypatch):
    """The nightlights lesson: 'checked, nothing found' and 'could not check' are different
    states, and neither is evidence of no flooding."""
    import varuna.serve.news as N
    store = N.NewsStore(root=str(tmp_path))
    assert store.coverage("patna") == "unknown"                          # never checked
    quiet, _ = N.classify(N.parse_rss(_rss(("Civic budget approved", "https://ex.com/q"))))
    store.add_items("patna", quiet)
    store.mark_fetch("patna", 1)
    assert store.coverage("patna") == "not_reported"                     # checked, nothing found
    wet, _ = N.classify(N.parse_rss(_rss(("Flood waters rise in Patna", "https://ex.com/w"))))
    store.add_items("patna", wet)
    store.mark_fetch("patna", 1)
    assert store.coverage("patna") == "reported"
    old = (_dt.datetime.now(_dt.timezone.utc)
           - _dt.timedelta(hours=N.STALE_AFTER_H + 1)).isoformat(timespec="seconds")
    store._fetch_ok["patna"] = old
    assert store.coverage("patna") == "unknown"                          # stale -> honest


def test_failed_fetch_is_not_marked_checked(tmp_path, monkeypatch):
    import varuna.serve.news as N
    monkeypatch.setattr(N, "_fetch_url",
                        lambda url, timeout=30: (_ for _ in ()).throw(RuntimeError("down")))
    store = N.NewsStore(root=str(tmp_path))
    out = N.update_news("patna", store=store)
    assert out["ok"] is False and out["coverage"] == "unknown"


# --------------------------------------------------------------------------- THE quarantine

def test_quarantine_burst_never_reaches_learner():
    """Adversarial: ten outlets cover one event — ten correlated points that would sail past
    the gate's evidence bar. The quarantine must strip every one before labelling."""
    from varuna.learn.news_labels import filter_reports_for_learning, news_to_labels
    from varuna.learn.labels import collect_labels
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    burst = [dict(id=f"n{i}", source="news", area="patna", ts=now, cell=[5, 5],
                  depth_m=0.4, depth_band="knee") for i in range(10)]
    citizens = [dict(id=f"c{i}", source="citizen", area="patna", ts=now, cell=[i + 1, i + 1],
                     depth_m=0.4, depth_band="knee") for i in range(2)]
    legacy = [dict(id="old1", area="patna", ts=now, cell=[9, 9],       # pre-`source` record
                   depth_m=0.8, depth_band="waist")]

    kept = filter_reports_for_learning(burst + citizens + legacy)
    assert {r["id"] for r in kept} == {"c0", "c1", "old1"}             # burst fully stripped

    days = collect_labels(kept, center=(25.6, 85.1), rain_fn=lambda d: 50.0)
    cells = {(r, c) for day in days for r, c, _m, _b in day["points"]}
    assert (5, 5) not in cells and len(cells) == 3                     # no news cell survives

    assert news_to_labels(burst) == []                                 # the hard boundary
