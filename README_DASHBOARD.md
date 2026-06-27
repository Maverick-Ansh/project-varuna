# Varuna Dashboard (React + Leaflet + FastAPI)

A live web UI for the Patna FloodTwin: an interactive map with toggleable layers (flood overlay,
ward alerts, sinks, recharge sites, canal plan), a **rainfall slider** that runs the U-Net emulator
live (milliseconds) to update the flood overlay, a SAR-validation panel (CSI textbook → calibrated),
canal planning, and an LLM chat box.

```
web/   React + Vite + react-leaflet front end
api/   FastAPI backend (serves the artifact bundle + live emulator/canal/optimize/chat)
```

## Backend

Needs the varuna package + its deps (torch, rasterio, numpy, pandas, matplotlib) for the live
emulator/canal endpoints. The light endpoints (layers, alerts, validation) work with just FastAPI.

```bash
pip install -r api/requirements.txt          # + the varuna build/serve deps
export VARUNA_WORK=artifacts/patna           # point at the committed Patna bundle
uvicorn api.server:app --reload --port 8000
```

Endpoints: `/api/health`, `/api/meta`, `/api/sinks`, `/api/recharge`, `/api/alerts`,
`/api/validation`, `/api/canal_plan`, `/api/image/{name}.png`, and POST `/api/whatif`,
`/api/canals`, `/api/optimize`, `/api/chat`. Heavy endpoints 503 cleanly if torch/rasterio
aren't installed, so the map still loads.

## Front end

```bash
cd web
npm install
npm run dev        # http://localhost:5173 ; proxies /api -> localhost:8000
# or: npm run build  (static bundle in web/dist for Vercel / GitHub Pages)
```

Set `VITE_API_BASE` (e.g. a deployed backend URL) for production; in dev the Vite proxy handles it.

## Deploy
- Backend: HF Spaces / Render (free). Mount the `artifacts/patna` bundle or set `VARUNA_WORK`.
- Front end: `npm run build` → Vercel / GitHub Pages, with `VITE_API_BASE` pointing at the backend.
