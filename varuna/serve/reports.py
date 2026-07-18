"""Citizen flood reports: live ingestion, map serving, and persistence.

People drop a pin + water level (ankle/knee/waist/chest) + optional note on the dashboard.
Reports are held in memory (the map + advisory read them instantly), appended to a per-boot
JSONL, and — when HF_TOKEN is set — mirrored to a Hugging Face *dataset* repo by a background
CommitScheduler (the free Space's filesystem is ephemeral; the dataset repo is the durable
store, and the nightly learner trains from it). On boot the store re-hydrates from the dataset
so restarts lose at most one scheduler window (~5 min) of reports.

Never writes into the artifact bundle dir (those writes are racy and vanish on redeploy).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import threading
import uuid

from ..config import CFG  # noqa: F401  (kept for parity with other serve modules)

log = logging.getLogger("varuna.serve.reports")

# Water level bands citizens can actually judge -> metres (band centre, Indian adult).
DEPTH_M = {"ankle": 0.1, "knee": 0.4, "waist": 0.8, "chest": 1.3}

# Fine-tune tolerance half-widths per band (the learner treats depths inside the band as
# zero-error; see varuna.learn). Kept here so the bands and their tolerances stay together.
DEPTH_TOL_M = {"ankle": 0.15, "knee": 0.20, "waist": 0.25, "chest": 0.35}

MAX_NOTE = 280
RATE_PER_IP_HOUR = 5           # reports per ip-hash per hour
RATE_PER_CELL_MIN = 30         # one report per (ip-hash, cell) per this many minutes
RATE_GLOBAL_DAY = 500          # hard global cap per UTC day


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(ts=None):
    return (ts or _utcnow()).isoformat(timespec="seconds")


def _parse_iso(s):
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def cell_of(work, lat, lon):
    """lat/lon -> twin-domain (row, col), or None when outside the N x N crop.

    Inverse of the crop mapping in build_domain: 30 m raster pixel via dem.tif's transform,
    minus the crop offset, /2 for the 60 m pooling. Needs torch+rasterio (present on the Space).
    """
    try:
        import torch
        import rasterio
        meta = torch.load(os.path.join(work, "twin_meta.pt"), map_location="cpu",
                          weights_only=False)
        with rasterio.open(os.path.join(work, "dem.tif")) as s:
            T = s.transform
        col30 = int((lon - T.c) / T.a)
        row30 = int((lat - T.f) / T.e)
        r = (row30 - int(meta["row0"])) // 2
        c = (col30 - int(meta["col0"])) // 2
        n = int(meta["n_grid"])
        if 0 <= r < n and 0 <= c < n:
            return int(r), int(c)
        return None
    except Exception as e:  # noqa: BLE001 - unbuilt bundle / missing deps -> not locatable
        log.warning("cell_of failed for %s: %s", work, e)
        return None


def sanitize_note(note):
    if not note:
        return ""
    clean = "".join(ch for ch in str(note) if ch.isprintable())
    return clean[:MAX_NOTE].strip()


class ReportStore:
    """Thread-safe in-memory + JSONL + (optional) HF-dataset-backed report store."""

    def __init__(self, root=None, dataset_id=None):
        self.root = root or os.environ.get("VARUNA_REPORTS_DIR", "/tmp/varuna_reports")
        self.dataset_id = dataset_id or os.environ.get("REPORTS_DATASET",
                                                       "AnshVivek/varuna-reports")
        self.boot_id = uuid.uuid4().hex[:8]
        month = _utcnow().strftime("%Y-%m")
        self.local_dir = os.path.join(self.root, "reports", month)
        os.makedirs(self.local_dir, exist_ok=True)
        self.path = os.path.join(self.local_dir, f"boot-{self.boot_id}.jsonl")
        self._lock = threading.Lock()
        self._reports = {}                # id -> report dict (includes ip_hash; never served raw)
        self._scheduler = None
        self._hydrated = False

    # ------------------------------------------------------------------ persistence

    def start(self):
        """Hydrate from the dataset repo and arm the background CommitScheduler (best-effort)."""
        threading.Thread(target=self._hydrate, daemon=True).start()
        token = os.environ.get("HF_TOKEN")
        if not token:
            log.info("reports: HF_TOKEN not set — reports persist only for this boot")
            return
        try:
            from huggingface_hub import CommitScheduler
            self._scheduler = CommitScheduler(
                repo_id=self.dataset_id, repo_type="dataset", token=token,
                folder_path=os.path.join(self.root, "reports"), path_in_repo="reports",
                every=5, private=False)
            log.info("reports: CommitScheduler -> %s every 5 min", self.dataset_id)
        except Exception as e:  # noqa: BLE001
            log.error("reports: CommitScheduler unavailable (%s) — boot-local only", e)

    def flush(self):
        if self._scheduler is not None:
            try:
                self._scheduler.trigger().result(timeout=60)
            except Exception as e:  # noqa: BLE001
                log.warning("reports: final flush failed: %s", e)

    def _hydrate(self):
        """Merge this + last month's JSONL from the dataset repo (and any local leftovers)."""
        try:
            files = []
            for d in (self.local_dir,):
                if os.path.isdir(d):
                    files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jsonl")]
            token = os.environ.get("HF_TOKEN")
            if token:
                from huggingface_hub import snapshot_download
                now = _utcnow()
                prev = (now.replace(day=1) - _dt.timedelta(days=1)).strftime("%Y-%m")
                snap = snapshot_download(
                    repo_id=self.dataset_id, repo_type="dataset", token=token,
                    allow_patterns=[f"reports/{now.strftime('%Y-%m')}/*",
                                    f"reports/{prev}/*"])
                for base, _dirs, fs in os.walk(os.path.join(snap, "reports")):
                    files += [os.path.join(base, f) for f in fs if f.endswith(".jsonl")]
            n = 0
            for path in files:
                if os.path.abspath(path) == os.path.abspath(self.path):
                    continue
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            rep = json.loads(line)
                            with self._lock:
                                self._reports.setdefault(rep["id"], rep)
                            n += 1
                        except Exception:  # noqa: BLE001
                            continue
            log.info("reports: hydrated %d reports from %d files", n, len(files))
        except Exception as e:  # noqa: BLE001
            log.warning("reports: hydration failed (fresh store): %s", e)
        finally:
            self._hydrated = True

    def _append_jsonl(self, rep):
        line = json.dumps(rep, ensure_ascii=False) + "\n"
        if self._scheduler is not None:
            with self._scheduler.lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------ rate limiting

    def _check_limits(self, ip_hash, cell, now):
        day_count, ip_count = 0, 0
        with self._lock:
            for rep in self._reports.values():
                try:
                    ts = _parse_iso(rep["ts"])
                except Exception:  # noqa: BLE001
                    continue
                age = (now - ts).total_seconds()
                if age < 0:
                    continue
                if age < 86400:
                    day_count += 1
                if rep.get("ip_hash") == ip_hash:
                    if age < 3600:
                        ip_count += 1
                    if (cell is not None and rep.get("cell") == list(cell)
                            and age < RATE_PER_CELL_MIN * 60):
                        raise ValueError("rate_limited: same spot reported minutes ago")
        if ip_count >= RATE_PER_IP_HOUR:
            raise ValueError("rate_limited: too many reports from this connection")
        if day_count >= RATE_GLOBAL_DAY:
            raise ValueError("rate_limited: global daily cap reached")

    # ------------------------------------------------------------------ API

    def add(self, area, work, lat, lon, depth_band, note="", client_ts=None, ip_hash=""):
        if depth_band not in DEPTH_M:
            raise ValueError(f"bad depth_band '{depth_band}'; one of {sorted(DEPTH_M)}")
        cell = cell_of(work, float(lat), float(lon))
        if cell is None:
            raise ValueError("outside_area: the pin is outside this area's model window")
        now = _utcnow()
        self._check_limits(ip_hash, cell, now)
        rep = dict(id=uuid.uuid4().hex[:12], source="citizen", area=area,
                   lat=round(float(lat), 6),
                   lon=round(float(lon), 6), depth_band=depth_band,
                   depth_m=DEPTH_M[depth_band], note=sanitize_note(note),
                   ts=_iso(now), client_ts=str(client_ts or "")[:32] or None,
                   cell=list(cell), ip_hash=ip_hash)
        with self._lock:
            self._reports[rep["id"]] = rep
        self._append_jsonl(rep)
        return self.public(rep)

    @staticmethod
    def public(rep):
        return {k: v for k, v in rep.items() if k != "ip_hash"}

    def recent(self, area=None, hours=24):
        cutoff = _utcnow() - _dt.timedelta(hours=float(hours))
        out = []
        with self._lock:
            for rep in self._reports.values():
                if area and rep.get("area") != area:
                    continue
                try:
                    if _parse_iso(rep["ts"]) < cutoff:
                        continue
                except Exception:  # noqa: BLE001
                    continue
                out.append(self.public(rep))
        out.sort(key=lambda r: r["ts"], reverse=True)
        return out

    def summary(self, area=None, hours=24):
        rs = self.recent(area=area, hours=hours)
        if not rs:
            return dict(count=0, max_depth_m=0.0, latest=None)
        return dict(count=len(rs), max_depth_m=max(r["depth_m"] for r in rs),
                    latest=rs[0]["ts"])


def ip_hash(ip, salt=None):
    salt = salt or os.environ.get("REPORT_SALT") or "varuna"
    return hashlib.sha256((salt + str(ip)).encode()).hexdigest()[:16]
