"""Nightly self-improvement + refresh job (scheduled Kaggle CPU notebook).

    reports (HF dataset) ──┐
    actual rain (Open-Meteo)├─> labels -> fine-tune candidate -> REWARD GATE ─┐
    replay buffer (bundle) ─┘                                                 │ accepted?
    Black Marble VNP46A2 (EE service account) -> outage layer                 ▼
    live-forecast alerts refresh                              ONE atomic Space commit

Source of truth for artifacts is the DEPLOYED SPACE (what users see), not git: the job
snapshots the Space, works on that copy, and pushes back exactly one commit (one rebuild).
Lineage (accepted checkpoints, gate records, label sets) goes to the reports dataset repo.

Secrets (Kaggle): HF_TOKEN (write: Space + dataset), EE_SERVICE_ACCOUNT_JSON (base64 of the
GCP key; optional -> --skip-ee), VARUNA_PROJECT_ID.

    python scripts/nightly_update.py --areas all [--dry-run] [--skip-ee] [--skip-train]
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import logging
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("nightly")

SPACE_ID = os.environ.get("VARUNA_SPACE", "AnshVivek/varuna-floodtwin")
DATASET_ID = os.environ.get("REPORTS_DATASET", "AnshVivek/varuna-reports")


def _token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception:  # noqa: BLE001
        return None


def _ee_key_path():
    """Materialise the EE service-account key from env/secret; None if unavailable."""
    raw = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if not raw:
        try:
            from kaggle_secrets import UserSecretsClient
            raw = UserSecretsClient().get_secret("EE_SERVICE_ACCOUNT_JSON")
        except Exception:  # noqa: BLE001
            return None
    try:
        decoded = base64.b64decode(raw).decode() if not raw.lstrip().startswith("{") else raw
        path = os.path.join(tempfile.gettempdir(), "ee_key.json")
        with open(path, "w") as f:
            f.write(decoded)
        return path
    except Exception as e:  # noqa: BLE001
        log.warning("EE key unusable: %s", e)
        return None


def fetch_state(token, workdir):
    """Space artifacts + all report JSONL -> local working copies."""
    from huggingface_hub import snapshot_download
    snap = snapshot_download(repo_id=SPACE_ID, repo_type="space", token=token,
                             allow_patterns=["artifacts/**"])
    art = os.path.join(workdir, "artifacts")
    shutil.copytree(os.path.join(snap, "artifacts"), art)
    reports = []
    try:
        rsnap = snapshot_download(repo_id=DATASET_ID, repo_type="dataset", token=token,
                                  allow_patterns=["reports/**"])
        for base, _d, files in os.walk(os.path.join(rsnap, "reports")):
            for f in files:
                if f.endswith(".jsonl"):
                    with open(os.path.join(base, f), encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                reports.append(json.loads(line))
                            except Exception:  # noqa: BLE001
                                pass
    except Exception as e:  # noqa: BLE001
        log.warning("no reports dataset yet (%s)", e)
    # dedupe by id
    reports = list({r["id"]: r for r in reports if r.get("id")}.values())
    log.info("state: artifacts -> %s | %d unique reports", art, len(reports))
    return art, reports


def learn_area(aid, work, reports, skip_train=False):
    """Label -> finetune -> gate for one area. Returns (changed, gate_entry|None)."""
    from varuna.areas import get_area
    from varuna.learn import collect_labels, split_holdout, finetune_emulator, gate, append_log

    mine = [r for r in reports if r.get("area") == aid]
    if not mine:
        log.info("%s: no reports — skip learning", aid)
        return False, None
    center = get_area(aid).center
    labels = collect_labels(mine, center)
    train, hold = split_holdout(labels)
    entry = dict(date=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                 accepted=False, n_reports=len(mine))
    if skip_train or not train or not hold:
        entry["reason"] = "skip_train flag" if skip_train else \
            f"insufficient storm-days ({len(labels)} labelled)"
        append_log(work, entry)
        return False, entry
    try:
        cand = finetune_emulator(work, train)
        entry = {**entry, **gate(work, cand, hold)}
    except Exception as e:  # noqa: BLE001
        entry["reason"] = f"learn failed: {e}"
        append_log(work, entry)
        return False, entry
    if entry["accepted"]:
        shutil.copy2(cand, os.path.join(work, "emulator.pt"))
        log.info("%s: candidate ACCEPTED (%s)", aid, entry["reason"])
    else:
        log.info("%s: candidate held back (%s)", aid, entry.get("reason"))
    if os.path.exists(os.path.join(work, "emulator_candidate.pt")):
        os.remove(os.path.join(work, "emulator_candidate.pt"))
    append_log(work, entry)
    return entry["accepted"], entry


def update_all_news(ids, token, dry_run):
    """Stage 2.5: fetch + triage news per area into the QUARANTINED news/ store.

    Runs after fetch_state and before any learning, and its output never joins `reports` —
    news is displayed, not trained on (varuna.learn.news_labels enforces the same boundary
    on the report list itself). One dataset commit for all areas, mirroring push_lineage.
    """
    from varuna.serve.news import NewsStore, update_news
    store = NewsStore(root=tempfile.mkdtemp(prefix="varuna_news_"))
    store.hydrate(token=token)                        # dedupe against what the dataset holds
    total_new = 0
    for aid in ids:
        try:
            total_new += update_news(aid, store=store)["new_items"]
        except Exception as e:  # noqa: BLE001
            log.warning("%s: news update failed: %s", aid, e)
    if dry_run or total_new == 0:
        log.info("news: %d new item(s)%s", total_new, " [dry-run — not pushed]" if dry_run else "")
        return total_new
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_folder(
            folder_path=os.path.join(store.root, "news"), repo_id=DATASET_ID,
            repo_type="dataset", path_in_repo="news",
            commit_message=f"nightly {_dt.date.today()}: {total_new} news item(s)")
        log.info("news: pushed %d new item(s) to %s", total_new, DATASET_ID)
    except Exception as e:  # noqa: BLE001
        log.warning("news: dataset push failed: %s", e)
    return total_new


def refresh_area(aid, work, ee_ready):
    """Non-learning refreshes: night lights + live alerts + (if model changed) ladder."""
    from varuna.areas import get_area
    changed = []
    if ee_ready:
        try:
            from varuna.build.nightlights import update_area
            update_area(work, aoi=list(get_area(aid).aoi))
            changed.append("nightlights")
        except Exception as e:  # noqa: BLE001
            log.warning("%s: night lights failed: %s", aid, e)
    try:
        from varuna.serve.alerts import run_alerts
        run_alerts(rain_mm=None, work=work, aoi=list(get_area(aid).aoi),
                   aggregate_wards=False, save_csv=True)
        changed.append("alerts")
    except Exception as e:  # noqa: BLE001
        log.warning("%s: alerts refresh failed: %s", aid, e)
    return changed


def push_space(token, artdir, changed_areas, summary, dry_run):
    if not changed_areas:
        log.info("nothing changed — no Space push")
        return
    if dry_run:
        log.info("[dry-run] would push %s to %s", sorted(changed_areas), SPACE_ID)
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_folder(folder_path=artdir, repo_id=SPACE_ID, repo_type="space",
                      path_in_repo="artifacts",
                      allow_patterns=[f"{aid}/**" for aid in sorted(changed_areas)],
                      commit_message=f"nightly {_dt.date.today()}: {summary}")
    log.info("pushed ONE commit to %s (%s)", SPACE_ID, summary)


def push_lineage(token, artdir, accepted, dry_run):
    if not accepted or dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    date = str(_dt.date.today())
    for aid in accepted:
        for fname in ("emulator.pt", "learning_log.json"):
            p = os.path.join(artdir, aid, fname)
            if os.path.exists(p):
                api.upload_file(path_or_fileobj=p, repo_id=DATASET_ID, repo_type="dataset",
                                path_in_repo=f"models/{aid}/{date}/{fname}",
                                commit_message=f"lineage {aid} {date}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+", default=["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-ee", action="store_true")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    token = _token()
    if not token:
        # A --dry-run pushes nothing: push_space, push_lineage and the news upload all return
        # before touching HfApi. The Space and the reports dataset are public, so the read half
        # works anonymously, and requiring a write token to exercise a no-write path is what
        # kept this job from ever being run at all. Refuse only for a real run.
        if not args.dry_run:
            raise SystemExit("HF_TOKEN missing (env or Kaggle secret)")
        log.warning("HF_TOKEN missing - continuing because --dry-run pushes nothing; "
                    "reads will be anonymous and any write would fail loudly")

    workdir = tempfile.mkdtemp(prefix="varuna_nightly_")
    artdir, reports = fetch_state(token, workdir)
    os.environ["VARUNA_ARTIFACTS"] = artdir           # point the registry at the working copy

    from varuna.areas import list_areas, is_built
    ids = ([a.id for a in list_areas() if is_built(a.id)]
           if args.areas == ["all"] else args.areas)

    # Quarantine gate: only citizen pins may become labels. News (stage 2.5 below) lives in
    # a parallel store and is never concatenated into `reports`.
    from varuna.learn.news_labels import filter_reports_for_learning
    reports = filter_reports_for_learning(reports)

    if not args.skip_news:
        update_all_news(ids, token, args.dry_run)

    ee_ready = False
    if not args.skip_ee:
        key = _ee_key_path()
        if key:
            try:
                from varuna.ee_auth import init_ee
                init_ee(os.environ.get("VARUNA_PROJECT_ID", "floodtwin"),
                        service_account_json=key)
                ee_ready = True
            except Exception as e:  # noqa: BLE001
                log.warning("EE init failed (%s) — night lights skipped", e)

    changed, accepted, decisions = set(), [], []
    for aid in ids:
        work = os.path.join(artdir, aid)
        ok, entry = learn_area(aid, work, reports, skip_train=args.skip_train)
        if entry:
            decisions.append((aid, entry.get("accepted"), entry.get("reason")))
            changed.add(aid)                          # learning_log.json always updates
        if ok:
            accepted.append(aid)
        if refresh_area(aid, work, ee_ready):
            changed.add(aid)

    summary = (f"{len(accepted)} model update(s) [{', '.join(accepted) or '-'}], "
               f"{len(changed)} area(s) refreshed")
    push_space(token, artdir, changed, summary, args.dry_run)
    push_lineage(token, artdir, accepted, args.dry_run)

    print("\n===== NIGHTLY SUMMARY =====")
    print(summary)
    for aid, acc, reason in decisions:
        print(f"  {aid:14s} {'ACCEPTED' if acc else 'held':8s} {reason}")
    print("===========================")


if __name__ == "__main__":
    main()
