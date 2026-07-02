# Launch runbook — Phase 3 (Colab) + deploy

Everything below is copy-paste ready. State at time of writing: `main` has the full
multi-area dashboard (Patna + 2 sub-crops built and committed; Bengaluru registered but
unbuilt), the calibrated/honest validation story (twin CSI 0.033), the canal optimizer
(20.2% cut) and storage sizing. Offline test suite: green.

## A. Colab session — exposure caches + Bengaluru build

Runtime: **GPU (T4)**. Internet: on.

```python
# Cell 1 — code + deps (torch is preinstalled on Colab)
!git clone https://github.com/Maverick-Ansh/project-varuna.git
%cd project-varuna
!pip -q install rasterio pysheds earthengine-api osmnx
```

```python
# Cell 2 — Earth Engine auth (needed ONLY for the Bengaluru build).
# Must be run BY YOU in the notebook tab (the auth bridge is user-only).
import ee
ee.Authenticate(auth_mode="notebook")
ee.Initialize(project="floodtwin")        # <- the GEE project id used for the Patna build
```

```python
# Cell 3 — exposure caches for the built Patna-family areas (no Earth Engine needed).
# Writes exposure.json into each bundle so the DEPLOYED dashboard (no osmnx) serves
# the exposure panel from cache instead of 503.
!python scripts/run_phase3_colab.py --exposure --rain 100
```

```python
# Cell 4 — Bengaluru: full EE build (sinks -> recharge -> twin) + all serve artifacts
# (alerts, canal plan, storage sizing, cost-benefit, maps, exposure). ~tens of minutes.
!python scripts/run_phase3_colab.py --build bengaluru --project-id floodtwin --rain 100
```

```python
# Cell 5 — push the bundles. Add a GITHUB_TOKEN secret first
# (Colab: key icon -> Secrets -> GITHUB_TOKEN, notebook access ON.
#  Kaggle: Add-ons -> Secrets — must be re-attached PER notebook).
!python scripts/run_phase3_colab.py --push --branch phase3-bengaluru \
    --message "Phase 3: exposure caches + Bengaluru bundle"
# then from the PC: gh pr create + merge (direct pushes to main from the PC are blocked).
```

Notes
- If the kernel restarts after `pip install`, just re-run cells 2 onward.
- Kaggle instead of Colab: work under `/kaggle/working`; secrets via `kaggle_secrets`
  (the script handles both automatically).
- The `--exposure` step alone needs no GPU and no EE — it can run first / separately.
- Forward+backward of the twin is ~10 s/date on a T4 — nothing here should "hang";
  a long silent step is most likely the EE image downloads.

## B. Deploy (user credentials required)

1. **Backend — Hugging Face Space (Docker):**
   ```bash
   HF_TOKEN=hf_xxx python deploy/deploy_hf_space.py Maverick-Ansh/varuna-floodtwin
   # smoke:
   curl https://maverick-ansh-varuna-floodtwin.hf.space/api/health
   ```
2. **Frontend — Vercel:** import the repo at vercel.com/new, Root Directory = `web`,
   env `VITE_API_BASE=https://maverick-ansh-varuna-floodtwin.hf.space`.
3. **Chat (optional):** add a free Groq API key as Space secret `LLM_API_KEY`
   (`LLM_API_BASE`/`LLM_MODEL` override the defaults in `api/chat_hosted.py`).

## C. Post-launch smoke checklist

- `GET /api/health` → ok, 4 areas; `GET /api/areas` → `bengaluru.built: true`.
- Dashboard: area picker recenters map; rainfall slider updates the flood overlay
  live for every area (per-area emulators).
- Exposure panel returns data everywhere (live where osmnx exists, `cached @ N mm` tag
  on the deployed Space).
- Validation panel shows the honest twin CSI (mean 0.033; best storm 2025-08-02 = 0.112).
- Canal planner: Patna ≈ 20.2% cut at 100 mm.

## D. Known deferred items

- SAR validation/calibration for Bengaluru (needs storm-date SAR masks; Patna-only for now).
- Real CGWB groundwater CSV (bundles carry the SAMPLE file).
- Paper writeup (workshop target: Climate Change AI / EGU).
