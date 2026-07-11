# Runbook — Varuna v2 "Live Mumbai"

What v2 adds: citizen flood reports (live map + nightly training labels), live-forecast mode,
water-budget panel ("where does the rain go?") with a storage-container designer, grounded
EN/HI/MR public advisories, NASA Black Marble power-outage layer, a reward-gated nightly
self-improvement loop, and full Mumbai (BMC) as four 256² tiles with a city-wide view.

## One-time setup (user actions)

1. **Reports dataset** — create `AnshVivek/varuna-reports` (Dataset, public) on Hugging Face.
2. **Space secrets** (Settings → Variables and secrets on `AnshVivek/varuna-floodtwin`):
   - `HF_TOKEN`: fine-grained token, write access to the reports dataset only.
   - `REPORT_SALT`: any random string (stabilises the report rate-limit hashes).
   - `LLM_API_KEY`: (already set) Groq key for chat + advisories.
3. **GEE service account** (for scheduled night-lights; skip → nightly uses `--skip-ee`):
   GCP console → project `floodtwin` → IAM → Service Accounts → create → grant
   "Earth Engine Resource Viewer" + register the account at code.earthengine.google.com/register
   → create JSON key → base64-encode (`base64 -w0 key.json`).
4. **Kaggle nightly** — upload `notebooks/08_nightly_update.ipynb` as a Kaggle notebook
   (CPU), attach secrets `HF_TOKEN`, `EE_SERVICE_ACCOUNT_JSON` (the base64), optional
   `VARUNA_PROJECT_ID` → Schedule → Daily. Verify the first run's NIGHTLY SUMMARY output.

## Mumbai builds (done 2026-07-11 on Colab 2×T4; repeat for new tiles)

```
!git clone -b <branch> https://github.com/Maverick-Ansh/project-varuna.git
%pip install -q rasterio pysheds earthengine-api osmnx
import ee; ee.Authenticate(auth_mode="notebook"); ee.Initialize(project="floodtwin")  # USER
!python scripts/run_phase3_colab.py --build mumbai_south --project-id floodtwin
# ... one per tile; ~15-25 min each on a T4. Post-build automatically writes: alerts,
# road graph (Overpass), spiderweb canal plan, storage sizing, cost-benefit, maps,
# exposure, water-balance ladder, gnn_data, replay buffer.
!python scripts/run_phase3_colab.py --push --branch <branch>   # GITHUB_TOKEN secret
```
Gotchas: don't start two builds at the same second (numba JIT cache race in pysheds —
stagger by ~a minute or run sequentially); a dead session resumes by re-running (built
tiles are skipped via `twin_meta.pt` guards).

Retrofit older bundles: `python scripts/make_replay.py --areas all --ladder`.

## Nightly loop (scripts/nightly_update.py)

reports (HF dataset) + actual rain (Open-Meteo) → labels (dry-day filter, per-cell median)
→ fine-tune candidate (replay-anchored, band-tolerant Huber) → **reward gate** (≥15 held-out
points from ≥2 storm-days, point error −5%, replay RMSE within +10%, bootstrap win ≥70%)
→ accepted candidates replace `emulator.pt` → night-lights refresh (VNP46A2, quality-masked,
missing≠dark) → live-alert CSVs → **one** `upload_folder` commit to the Space (one rebuild)
→ lineage (checkpoints + learning_log) to the dataset repo. Flags: `--dry-run --skip-ee
--skip-train --areas <ids>`. Every decision lands in `learning_log.json` (dashboard shows it).

## Deploy

- Backend: `HF_TOKEN=… python deploy/deploy_hf_space.py AnshVivek/varuna-floodtwin`
  (or merge to main and run it from Kaggle as before).
- Frontend: Vercel auto-rebuilds on main; new areas appear automatically via /api/areas.

## Smoke checklist (after deploy)

- `/api/health` → 8 areas; `/api/weather?area=mumbai_west` → real mm; `/api/alerts_live` 200.
- POST `/api/reports` (pin inside the tile) → 200; marker on the dashboard within 60 s;
  commit appears in the reports dataset within ~5 min; survives a Space restart.
- `/api/waterbalance?area=mumbai_south&rain_mm=100` → closure sums; panel bar moves with slider.
- `/api/advisory?area=mumbai_west` → en+hi+mr; `/api/nightlights?area=patna` → date + cells.
- `/api/city?city=mumbai` → 4 tiles, totals; dashboard "Mumbai (city)" view renders tiles.
- Route + canals still work per tile (regression).
