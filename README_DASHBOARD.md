# Varuna Dashboard (React + Leaflet + FastAPI)

A live web UI for the FloodTwin: an **area picker** (Patna + any other built area), an interactive
map with toggleable layers (flood overlay, ward alerts, sinks, recharge, canals, dig sites, at-risk
buildings, evacuation roads), a **rainfall slider** that runs the per-area U-Net emulator live
(milliseconds) to update the flood overlay, intervention buttons (canals / storage / excavation), a
**cost-benefit** ranking (₹ per m³ of flood removed), an **exposure/evacuation** view, a one-click
**AI plan report**, a SAR-validation panel, and an LLM chat box — all scoped to the selected area.

```
web/   React + Vite + react-leaflet front end
api/   FastAPI backend (serves the artifact bundle + live emulator/canal/optimize/chat)
```

## Backend

Needs the varuna package + its deps (torch, rasterio, numpy, pandas, matplotlib) for the live
emulator/canal endpoints. The light endpoints (layers, alerts, validation) work with just FastAPI.

```bash
pip install -r api/requirements.txt          # + the varuna build/serve deps
uvicorn api.server:app --reload --port 8000
# areas come from varuna/areas.py; no VARUNA_WORK needed (it defaults to the first built area).
# VARUNA_WORK still works as a fallback bundle dir when no ?area= is given.
```

Every endpoint takes an optional `area` id (`?area=` for GET, `"area"` in the POST body) and
serves that area's bundle. Endpoints: `/api/areas`, `/api/health`, `/api/meta`, `/api/sinks`,
`/api/recharge`, `/api/alerts`, `/api/validation`, `/api/canal_plan`, `/api/image/{name}.png`,
and POST `/api/whatif`, `/api/canals`, `/api/storage`, `/api/optimize`, `/api/costbenefit`,
`/api/exposure`, `/api/report`, `/api/chat`. Heavy endpoints 503 cleanly if their deps
(torch/rasterio/osmnx) aren't installed, so the map still loads.

## Front end

```bash
cd web
npm install
npm run dev        # http://localhost:5173 ; proxies /api -> localhost:8000
# or: npm run build  (static bundle in web/dist for Vercel / GitHub Pages)
```

Set `VITE_API_BASE` (e.g. a deployed backend URL) for production; in dev the Vite proxy handles it.

## Multi-area

Areas are registered in `varuna/areas.py` (`{id, name, aoi, center, work_dir}`); each has its own
bundle under `artifacts/<id>/`. The serve layer is area-correct from the bundle alone —
`build_domain` reads the crop centre/grid from the bundle's `twin_meta.pt`, so pointing an endpoint
at a different bundle "just works". Build a new area's bundle with:

```bash
python -m varuna build --area patna_east        # sub-crop: reuses Patna's DEM, no Earth Engine
python -m varuna build --area bengaluru --project-id <EE_PROJECT>   # full build (EE + GPU for the twin)
```

Sub-crops (`source_work` set in the registry) reuse a source area's downloaded rasters and only
retrain the twin at a new centre — so they need no Earth Engine and build on CPU. Each area gets its
**own U-Net emulator** so the live slider stays instant everywhere.

> Emulator note: `train_emulator` up-weights wet cells (flooding is sparse, so a plain full-grid MSE
> collapses to predicting "no flood"). The reported flooded-cell RMSE is the honest skill metric.

## Deploy
- Backend: HF Spaces / Render (free). Ships the `artifacts/*` bundles; areas appear automatically.
- Front end: `npm run build` → Vercel / GitHub Pages, with `VITE_API_BASE` pointing at the backend.
