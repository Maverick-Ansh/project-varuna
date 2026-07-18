"""Public flood news: ingest, extract, display — and QUARANTINE from the learner.

The quarantine is the design, not a limitation. News volume tracks media attention, not
water depth: ten outlets covering one Bengaluru flood are ten *correlated* points, and the
reward gate's bootstrap (varuna.learn.reward) assumes i.i.d. evidence. In V3 news is stored
and displayed, never trained on — `varuna.learn.news_labels` is the enforcement boundary.

House patterns this module follows deliberately:
  * advisory.py — the deterministic path (keyword triage) is computed FIRST; the hosted LLM
    only refines it, `except Exception` falls back, and `backend: hosted|template` ships in
    the payload. The LLM does language; Python does every decision.
  * roadnet.py — multiple feed providers, a real User-Agent, warn-per-failure, and the cached
    store as the fallback when every provider is down.
  * nightlights.py — the three-state model. No news about a flood is NOT evidence of no
    flood: `coverage` is `reported` / `not_reported` (fresh fetch, nothing found) / `unknown`
    (could not check within the staleness ceiling), and the caveat ships inside the payload.
  * reports.py — same JSONL + CommitScheduler persistence, but a PARALLEL store under the
    `news/` prefix: news must never flow through ReportStore.add (its rate-limit scan is
    sized for 500 citizen pins/day, and mixing would blur provenance forever).

Every record carries `source: "news"` from the very first write — provenance is the one
decision that is expensive to reverse.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
import threading
import uuid
import xml.etree.ElementTree as _ET
from email.utils import parsedate_to_datetime

from .reports import DEPTH_M

log = logging.getLogger("varuna.serve.news")

DEFAULT_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Feed providers (the roadnet "mirrors" idiom — independent providers, same role).
FEED_PROVIDERS = (
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en",
    "https://www.bing.com/news/search?q={query}&format=rss",
)
USER_AGENT = "varuna-floodtwin/1.0 (github.com/Maverick-Ansh/project-varuna)"

# Past this, a missing fetch means "could not check", not "nothing happened".
STALE_AFTER_H = 30.0

QUARANTINE_NOTE = (
    "News tracks media attention, not water depth — absence of coverage is NOT evidence of "
    "no flooding (coverage distinguishes 'checked, nothing found' from 'could not check'). "
    "Items are stored and displayed only; they are NEVER used to train the model."
)

_FLOOD_WORDS = ("flood", "waterlog", "water-log", "inundat", "submerg", "deluge",
                "cloudburst", "downpour", "rains lash", "heavy rain", "जलभराव", "बाढ़")

# Depth phrases citizens/journalists actually write -> the exact band centres the learner's
# reverse lookup expects (labels.py matches DEPTH_M by float equality — snap, never eyeball).
_BAND_WORDS = {"ankle": "ankle", "knee": "knee", "waist": "waist", "hip": "waist",
               "chest": "chest", "neck": "chest", "shoulder": "chest"}
_FEET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:feet|foot|ft)\b", re.I)
_METRE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:metre|meter)s?\b", re.I)


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(ts=None):
    return (ts or _utcnow()).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- fetch + parse

def _fetch_url(url, timeout=30):
    """GET one feed URL -> raw bytes. Module-level so tests monkeypatch the symbol."""
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content


def parse_rss(raw):
    """RSS bytes -> [{title, url, outlet, published}]. Defensive: skip malformed items."""
    items = []
    root = _ET.fromstring(raw)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        link = (it.findtext("link") or "").strip()
        outlet = (it.findtext("source") or "").strip()
        if not outlet and link:
            m = re.search(r"https?://(?:www\.)?([^/]+)/", link + "/")
            outlet = m.group(1) if m else ""
        published = None
        pd = it.findtext("pubDate")
        if pd:
            try:
                published = _iso(parsedate_to_datetime(pd).astimezone(_dt.timezone.utc))
            except Exception:  # noqa: BLE001
                published = None
        items.append(dict(title=title, url=link, outlet=outlet, published=published))
    return items


def fetch_news(query, timeout=30, providers=FEED_PROVIDERS):
    """Query every provider until one answers. Raises after the last — caller keeps its cache."""
    from urllib.parse import quote_plus
    last = None
    for tpl in providers:
        url = tpl.format(query=quote_plus(query))
        try:
            items = parse_rss(_fetch_url(url, timeout=timeout))
            log.info("news: %d items from %s", len(items), url.split("/")[2])
            return items
        except Exception as exc:  # noqa: BLE001
            log.warning("news provider failed (%s): %s", url.split("/")[2], exc)
            last = exc
    raise RuntimeError(f"all news providers failed (last: {last}); serving cached items only")


# --------------------------------------------------------------------------- triage

def flood_related(text):
    """Deterministic keyword triage — the fallback that always exists."""
    t = str(text).lower()
    return any(w in t for w in _FLOOD_WORDS)


def snap_depth(text):
    """Depth phrase -> (band, exact band-centre metres) or (None, None).

    labels.py reverse-looks-up the band by float equality on DEPTH_M values, so anything we
    emit MUST be an exact band centre — a raw '0.9 m from 3 feet' would silently mislabel.
    """
    t = str(text or "").lower()
    for word, band in _BAND_WORDS.items():
        if word in t:
            return band, DEPTH_M[band]
    m = _FEET_RE.search(t) or _METRE_RE.search(t)
    if m:
        metres = float(m.group(1)) * (0.3048 if _FEET_RE.search(t) else 1.0)
        band = min(DEPTH_M, key=lambda b: abs(DEPTH_M[b] - metres))
        return band, DEPTH_M[band]
    return None, None


_LLM_SYSTEM = (
    "You triage Indian news headlines about urban flooding. For each numbered headline "
    "decide if it reports CURRENT flooding/waterlogging (not politics, budgets or history), "
    "and copy out any location phrase and any water-depth phrase VERBATIM from the headline "
    "— never invent or infer numbers or places that are not in the text. Output STRICT "
    'JSON: {"items": [{"i": <headline number>, "flood": true|false, '
    '"location": "<verbatim or empty>", "depth": "<verbatim or empty>"}]}.'
)


def _llm_call(messages, timeout=30.0):
    """One chat call, advisory-style. Module-level so tests monkeypatch the symbol."""
    import requests
    key = os.environ["LLM_API_KEY"]
    base = os.environ.get("LLM_API_BASE", DEFAULT_BASE).rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    r = requests.post(f"{base}/chat/completions",
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      json={"model": model, "messages": messages, "temperature": 0.2,
                            "max_tokens": 1200, "response_format": {"type": "json_object"}},
                      timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def classify(items, timeout=30.0):
    """Raw items -> triaged items + backend. Keyword pass FIRST; the LLM only refines.

    Python makes every decision: an item is flood-related if the keywords OR the LLM say so
    (both flags are stored for audit), and depth always goes through snap_depth — the LLM
    never emits a number we trust directly.
    """
    out = []
    for it in items:
        kw = flood_related(it["title"])
        band, depth = snap_depth(it["title"])
        out.append({**it, "flood_related": kw, "matched": "keywords" if kw else "",
                    "location_text": "", "depth_band": band, "depth_m": depth})
    backend = "template"
    if os.environ.get("LLM_API_KEY") and items:
        try:
            lines = "\n".join(f"{i}. {it['title']}" for i, it in enumerate(items))
            txt = _llm_call([{"role": "system", "content": _LLM_SYSTEM},
                             {"role": "user", "content": "Headlines:\n" + lines}],
                            timeout=timeout)
            for row in json.loads(txt)["items"]:
                i = int(row["i"])
                if not 0 <= i < len(out):
                    continue
                if row.get("flood"):
                    out[i]["flood_related"] = True
                    out[i]["matched"] = "both" if out[i]["matched"] else "llm"
                out[i]["location_text"] = str(row.get("location") or "")[:120]
                if out[i]["depth_band"] is None and row.get("depth"):
                    band, depth = snap_depth(row["depth"])      # Python snaps, LLM only quotes
                    out[i]["depth_band"], out[i]["depth_m"] = band, depth
            backend = "hosted"
        except Exception as e:  # noqa: BLE001
            log.warning("hosted news triage failed (%s) — keyword fallback", e)
    return out, backend


# --------------------------------------------------------------------------- the store

class NewsStore:
    """JSONL + (optional) HF-dataset store for news items — PARALLEL to ReportStore.

    Same persistence idiom (per-boot JSONL, CommitScheduler, hydrate-on-start) but under the
    `news/` prefix and without citizen rate limiting: this is our own fetch, not public
    input. Records are deduped by a stable id (hash of url|title) so nightly re-fetches
    never duplicate. `kind: "fetch_ok"` marker rows record that a fetch SUCCEEDED even when
    it found nothing — that is what lets `not_reported` and `unknown` stay distinct states.
    """

    def __init__(self, root=None, dataset_id=None):
        self.root = root or os.environ.get("VARUNA_REPORTS_DIR", "/tmp/varuna_reports")
        self.dataset_id = dataset_id or os.environ.get("REPORTS_DATASET",
                                                       "AnshVivek/varuna-reports")
        self.boot_id = uuid.uuid4().hex[:8]
        month = _utcnow().strftime("%Y-%m")
        self.local_dir = os.path.join(self.root, "news", month)
        os.makedirs(self.local_dir, exist_ok=True)
        self.path = os.path.join(self.local_dir, f"boot-{self.boot_id}.jsonl")
        self._lock = threading.Lock()
        self._items = {}                    # id -> item dict
        self._fetch_ok = {}                 # area -> latest successful-fetch iso ts
        self._scheduler = None

    # ------------------------------------------------------------------ persistence

    def start(self):
        """Async hydrate + arm the CommitScheduler (Space serving path; best-effort)."""
        threading.Thread(target=self.hydrate, daemon=True).start()
        token = os.environ.get("HF_TOKEN")
        if not token:
            log.info("news: HF_TOKEN not set — items persist only for this boot")
            return
        try:
            from huggingface_hub import CommitScheduler
            self._scheduler = CommitScheduler(
                repo_id=self.dataset_id, repo_type="dataset", token=token,
                folder_path=os.path.join(self.root, "news"), path_in_repo="news",
                every=5, private=False)
            log.info("news: CommitScheduler -> %s every 5 min", self.dataset_id)
        except Exception as e:  # noqa: BLE001
            log.error("news: CommitScheduler unavailable (%s) — boot-local only", e)

    def flush(self):
        if self._scheduler is not None:
            try:
                self._scheduler.trigger().result(timeout=60)
            except Exception as e:  # noqa: BLE001
                log.warning("news: final flush failed: %s", e)

    def hydrate(self, token=None):
        """Merge this + last month's news JSONL from disk and (with a token) the dataset."""
        try:
            files = []
            if os.path.isdir(self.local_dir):
                files += [os.path.join(self.local_dir, f)
                          for f in os.listdir(self.local_dir) if f.endswith(".jsonl")]
            token = token or os.environ.get("HF_TOKEN")
            if token:
                from huggingface_hub import snapshot_download
                now = _utcnow()
                prev = (now.replace(day=1) - _dt.timedelta(days=1)).strftime("%Y-%m")
                snap = snapshot_download(
                    repo_id=self.dataset_id, repo_type="dataset", token=token,
                    allow_patterns=[f"news/{now.strftime('%Y-%m')}/*", f"news/{prev}/*"])
                for base, _dirs, fs in os.walk(os.path.join(snap, "news")):
                    files += [os.path.join(base, f) for f in fs if f.endswith(".jsonl")]
            n = 0
            for path in files:
                if os.path.abspath(path) == os.path.abspath(self.path):
                    continue
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            self._absorb(json.loads(line))
                            n += 1
                        except Exception:  # noqa: BLE001
                            continue
            log.info("news: hydrated %d records from %d files", n, len(files))
        except Exception as e:  # noqa: BLE001
            log.warning("news: hydration failed (fresh store): %s", e)

    def _absorb(self, rec):
        with self._lock:
            if rec.get("kind") == "fetch_ok":
                prev = self._fetch_ok.get(rec["area"], "")
                self._fetch_ok[rec["area"]] = max(prev, rec.get("fetched_at") or "")
            elif rec.get("id"):
                self._items.setdefault(rec["id"], rec)

    def _append_jsonl(self, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        if self._scheduler is not None:
            with self._scheduler.lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------ API

    def add_items(self, area, items, backend="template"):
        """Store triaged items (deduped). Every record leaves here with source='news'."""
        n_new = 0
        now = _iso()
        for it in items:
            iid = hashlib.sha256((it.get("url") or it["title"]).encode()).hexdigest()[:12]
            with self._lock:
                if iid in self._items:
                    continue
            rec = dict(id=iid, source="news", area=area, title=it["title"][:300],
                       url=it.get("url") or "", outlet=it.get("outlet") or "",
                       ts=it.get("published") or now, fetched_at=now,
                       flood_related=bool(it.get("flood_related")),
                       matched=it.get("matched") or "",
                       location_text=it.get("location_text") or "",
                       depth_band=it.get("depth_band"), depth_m=it.get("depth_m"),
                       cell=None, backend=backend)
            with self._lock:
                self._items[iid] = rec
            self._append_jsonl(rec)
            n_new += 1
        return n_new

    def mark_fetch(self, area, n_items):
        rec = dict(kind="fetch_ok", area=area, n_items=int(n_items), fetched_at=_iso())
        self._absorb(rec)
        self._append_jsonl(rec)

    def recent(self, area, hours=24, flood_only=True):
        cutoff = _iso(_utcnow() - _dt.timedelta(hours=float(hours)))
        with self._lock:
            rows = [r for r in self._items.values()
                    if r.get("area") == area and (r.get("ts") or r["fetched_at"]) >= cutoff
                    and (r.get("flood_related") or not flood_only)]
        rows.sort(key=lambda r: r.get("ts") or r["fetched_at"], reverse=True)
        return rows

    def coverage(self, area):
        """Three-state: reported / not_reported / unknown (the nightlights lesson)."""
        last = self._fetch_ok.get(area)
        stale_at = _iso(_utcnow() - _dt.timedelta(hours=STALE_AFTER_H))
        if not last or last < stale_at:
            return "unknown"
        return "reported" if self.recent(area, hours=STALE_AFTER_H) else "not_reported"

    def feed(self, area, hours=24):
        """The servable payload — items + coverage + the quarantine caveat, always inside."""
        items = self.recent(area, hours=hours)
        return dict(area=area, hours=float(hours), count=len(items), items=items,
                    coverage=self.coverage(area),
                    last_fetch=self._fetch_ok.get(area),
                    trained_on=False, note=QUARANTINE_NOTE)


# --------------------------------------------------------------------------- driver

def area_query(area_name):
    """'Mumbai — Island City' / 'Patna (greater)' -> the city word the feeds understand."""
    name = re.split(r"[—(-]", str(area_name))[0].strip()
    return f"{name or area_name} flood waterlogging"


def update_news(area_id, store, timeout=30.0):
    """Fetch + triage + store one area's news. A failed fetch is NOT marked as checked —
    that is exactly what keeps `unknown` honest."""
    from ..areas import get_area
    area = get_area(area_id)
    try:
        raw = fetch_news(area_query(area.name), timeout=timeout)
    except Exception as e:  # noqa: BLE001
        log.warning("news fetch for '%s' failed: %s — cache only", area_id, e)
        return dict(area=area_id, ok=False, new_items=0, coverage=store.coverage(area_id))
    triaged, backend = classify(raw, timeout=timeout)
    n_new = store.add_items(area_id, triaged, backend=backend)
    store.mark_fetch(area_id, len(triaged))
    log.info("news '%s': %d items (%d new, %d flood-related, backend=%s)", area_id,
             len(triaged), n_new, sum(t["flood_related"] for t in triaged), backend)
    return dict(area=area_id, ok=True, new_items=n_new, backend=backend,
                coverage=store.coverage(area_id))
