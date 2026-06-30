# Deploying the Varuna dashboard

Backend → **Hugging Face Spaces** (Docker, free CPU). Frontend → **Vercel**. Chat → a **free hosted
LLM API** (Groq by default). No GPU required.

## 1. Backend — HF Space (Docker)

The Space is built from this repo's `Dockerfile` (CPU torch + rasterio so the live `/whatif`,
`/canals`, `/optimize` endpoints work, not just the static bundle).

1. Create a write token at https://huggingface.co/settings/tokens and a new **Docker** Space
   (e.g. `your-name/varuna-floodtwin`).
2. Push the backend to it. From the repo root:
   ```bash
   # one-time: stage only what the Space needs (Dockerfile, requirements-deploy.txt, varuna/, api/, artifacts/)
   git clone https://huggingface.co/spaces/<you>/varuna-floodtwin hf-space && cd hf-space
   cp ../Dockerfile ../requirements-deploy.txt .
   cp ../deploy/space_README.md ./README.md      # HF reads the YAML front-matter here
   cp -r ../varuna ../api ../artifacts .
   rm -f artifacts/patna/observed_water_*.tif artifacts/patna/twin_dataset.pt   # keep image lean
   git add -A && git commit -m "Varuna FloodTwin backend" && git push
   ```
   (Authenticate the push with your HF token as the password, user = your HF username.)
3. The Space builds and serves on port 7860. Smoke-test:
   `https://<you>-varuna-floodtwin.hf.space/api/health` → `{"ok": true, ...}`.
4. **Chat (optional, free):** get a free key at https://console.groq.com/keys and add Space secrets
   `LLM_API_KEY` (and optionally `LLM_API_BASE`, `LLM_MODEL`). The `/api/chat` endpoint uses it; if
   unset it 503s gracefully and the rest of the dashboard works.

## 2. Frontend — Vercel

`web/` is a Vite + react-leaflet SPA that reads the API base from `VITE_API_BASE`.

1. Import the GitHub repo at https://vercel.com/new and set **Root Directory = `web`** (Vercel reads
   `web/vercel.json`; framework auto-detects Vite).
2. Add an env var **`VITE_API_BASE`** = your Space URL, e.g.
   `https://<you>-varuna-floodtwin.hf.space` (no trailing slash).
3. Deploy. The CORS on the API is open (`*`), so the browser can call the Space directly.

CLI alternative:
```bash
cd web && npm install && npm run build
npx vercel --prod -e VITE_API_BASE="https://<you>-varuna-floodtwin.hf.space"
```

## 3. Smoke test end to end
- `GET /api/health`, `/api/meta`, `/api/validation` return JSON.
- Frontend loads the map; the rainfall slider hits `POST /api/whatif` and the flood overlay updates.
- (If `LLM_API_KEY` set) the chat panel answers and reports `"backend": "hosted"`.
