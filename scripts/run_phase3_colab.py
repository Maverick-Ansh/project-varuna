"""Colab/Kaggle driver for Phase 3: exposure caching, new-area builds, bundle push.

Three independent jobs (combine flags freely):

  --exposure            compute + cache exposure.json (OSM buildings/roads vs flood depth)
                        for every built area (or --areas ...). Needs osmnx; NO Earth Engine.
  --build <area_id>     build a registered area's bundle (see varuna.areas). Sub-crops are
                        CPU-only; full areas (bengaluru) need Earth Engine auth + GPU.
                        Afterwards writes the serve artifacts the dashboard reads
                        (alerts, canal plan, storage sizing, cost-benefit, maps, exposure).
  --push                commit every artifacts/<area> bundle to git and push (force-adds
                        the gitignored .tif/.pt/.json artifacts). Token comes from the
                        GITHUB_TOKEN env var, Colab secret, or Kaggle secret — never printed.

Colab (GPU runtime) — full Phase 3:

    !git clone https://github.com/Maverick-Ansh/project-varuna.git
    %cd project-varuna
    !pip -q install rasterio pysheds earthengine-api osmnx
    # Earth Engine auth (ONLY needed for --build of a full area) — run yourself:
    import ee; ee.Authenticate(auth_mode='notebook'); ee.Initialize(project='floodtwin')
    !python scripts/run_phase3_colab.py --exposure
    !python scripts/run_phase3_colab.py --build bengaluru --project-id floodtwin
    !python scripts/run_phase3_colab.py --push

Kaggle: same, but secrets come from kaggle_secrets (attach GITHUB_TOKEN under
Add-ons -> Secrets, per notebook) and the work dir is /kaggle/working.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
log = logging.getLogger("run_phase3")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ exposure

def do_exposure(area_ids, rain_mm):
    from varuna.areas import list_areas, get_area, is_built
    from varuna.serve.exposure import save_exposure
    ids = area_ids or [a.id for a in list_areas() if is_built(a.id)]
    done = {}
    for aid in ids:
        if not is_built(aid):
            log.warning("area '%s' has no built bundle — skipping exposure", aid)
            continue
        work = get_area(aid).work_dir()
        log.info("exposure for '%s' at %.0f mm (work=%s)", aid, rain_mm, work)
        try:
            out = save_exposure(rain_mm=rain_mm, work=work)
            done[aid] = dict(buildings_at_risk=out["buildings"]["at_risk"],
                             buildings_total=out["buildings"]["total_in_domain"],
                             roads_flooded=out["roads"]["flooded"],
                             roads_total=out["roads"]["total"])
        except Exception as e:  # noqa: BLE001
            log.error("exposure for '%s' failed: %s", aid, e)
    return done


# ------------------------------------------------------------------ build

def _post_build_artifacts(area, rain_mm):
    """Write the serve-layer artifacts the dashboard reads. Each step is best-effort."""
    work = area.work_dir()

    def step(name, fn):
        try:
            fn()
            log.info("post-build %-12s OK", name)
        except Exception as e:  # noqa: BLE001
            log.error("post-build %-12s FAILED: %s", name, e)

    def alerts():
        from varuna.serve.alerts import run_alerts
        try:
            run_alerts(work=work, aoi=list(area.aoi))
        except Exception:  # ward aggregation needs geopandas/OSM — retry without
            run_alerts(work=work, aoi=list(area.aoi), aggregate_wards=False)

    def canals():
        from varuna.serve.canals import plan_canals
        r = plan_canals(rain_mm=rain_mm, work=work)
        log.info("  canal plan: %.1f%% cut", r.get("reduction_pct", float("nan")))

    def storage():
        from varuna.serve.containers import plan_storage, plot_storage_dose
        rep = plan_storage(rain_mm=rain_mm, work=work)
        figdir = os.path.join(work, "figures")
        os.makedirs(figdir, exist_ok=True)
        plot_storage_dose(rep, out=os.path.join(figdir, "storage_dose.png"))

    def costbenefit():
        from varuna.serve.costbenefit import rank_interventions
        rank_interventions(rain_mm=rain_mm, work=work)

    def maps():
        from varuna.viz import render_all
        render_all(work=work, rain_mm=rain_mm)

    def exposure():
        from varuna.serve.exposure import save_exposure
        save_exposure(rain_mm=rain_mm, work=work)

    step("alerts", alerts)
    step("canals", canals)
    step("storage", storage)
    step("costbenefit", costbenefit)
    step("maps", maps)
    step("exposure", exposure)


def do_build(area_id, project_id, rain_mm, n_samples=None, epochs=40, skip_artifacts=False):
    from varuna.areas import get_area
    from varuna.build.areas_build import build_area
    area = get_area(area_id)
    if not area.source_work:                       # full build -> Earth Engine
        from varuna.ee_auth import init_ee
        init_ee(project_id)
    build_area(area_id, project_id=project_id, n_samples=n_samples, epochs=epochs)
    if not skip_artifacts:
        _post_build_artifacts(area, rain_mm)
    log.info("build for '%s' complete -> %s", area_id, area.work_dir())


# ------------------------------------------------------------------ push

def _github_token():
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    try:                                            # Colab secret
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN").strip()
    except Exception:  # noqa: BLE001
        pass
    try:                                            # Kaggle secret (attach per notebook!)
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _git(*args, token=None, check=True):
    """Run git, scrubbing the token from anything we print or raise."""
    r = subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if token:
        out = out.replace(token, "***")
    if r.returncode != 0 and check:
        raise RuntimeError(f"git {args[0]} failed: {out}")
    return out


def do_push(message, branch=None):
    tok = _github_token()
    if not tok:
        raise SystemExit("no GITHUB_TOKEN (env var / Colab secret / Kaggle secret). "
                         "On Kaggle, attach the secret to THIS notebook (Add-ons -> Secrets).")
    if not _git("config", "user.email", check=False):
        _git("config", "user.email", "colab@varuna.local")
        _git("config", "user.name", "Varuna Colab")

    from varuna.areas import list_areas, is_built
    added = []
    for a in list_areas():
        if is_built(a.id):
            rel = os.path.relpath(a.work_dir(), REPO_ROOT)
            if not rel.startswith(".."):
                _git("add", "-f", rel)              # bundle files are gitignored -> force
                added.append(rel)
    if not _git("status", "--porcelain", check=False):
        log.info("nothing to push — bundles unchanged")
        return
    _git("commit", "-m", message)

    remote = _git("remote", "get-url", "origin")
    slug = remote.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    url = f"https://x-access-token:{tok}@github.com/{slug}.git"
    branch = branch or _git("rev-parse", "--abbrev-ref", "HEAD")
    out = _git("push", url, f"HEAD:{branch}", token=tok)
    log.info("pushed %s -> %s: %s", added, branch, out or "ok")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Varuna Phase 3 (Colab/Kaggle)")
    ap.add_argument("--exposure", action="store_true", help="cache exposure.json for built areas")
    ap.add_argument("--areas", nargs="+", default=None, help="area ids for --exposure")
    ap.add_argument("--build", default=None, metavar="AREA_ID",
                    help="build this registered area (bengaluru needs EE auth + GPU)")
    ap.add_argument("--project-id", default=os.environ.get("VARUNA_PROJECT_ID"),
                    help="Earth Engine project id (full builds)")
    ap.add_argument("--rain", type=float, default=100.0,
                    help="design storm (mm) for exposure/canals/storage artifacts")
    ap.add_argument("--n-samples", type=int, default=None, help="twin training samples")
    ap.add_argument("--epochs", type=int, default=40, help="twin training epochs")
    ap.add_argument("--skip-artifacts", action="store_true",
                    help="with --build: skip the post-build serve artifacts")
    ap.add_argument("--push", action="store_true", help="commit + push built bundles")
    ap.add_argument("--message", default="Phase 3: area bundles (exposure/build) from Colab",
                    help="commit message for --push")
    ap.add_argument("--branch", default=None, help="push target branch (default: current)")
    args = ap.parse_args()

    if not (args.exposure or args.build or args.push):
        raise SystemExit("nothing to do: pass --exposure and/or --build <area> and/or --push")

    if args.build:
        do_build(args.build, args.project_id, args.rain,
                 n_samples=args.n_samples, epochs=args.epochs,
                 skip_artifacts=args.skip_artifacts)
    if args.exposure:
        summary = do_exposure(args.areas, args.rain)
        print("\nEXPOSURE SUMMARY")
        for aid, s in summary.items():
            print(f"  {aid:12s} buildings at risk {s['buildings_at_risk']}/{s['buildings_total']}"
                  f"  roads flooded {s['roads_flooded']}/{s['roads_total']}")
    if args.push:
        do_push(args.message, branch=args.branch)


if __name__ == "__main__":
    sys.path.insert(0, REPO_ROOT)
    main()
