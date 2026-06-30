---
title: Varuna FloodTwin Patna API
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Varuna FloodTwin — Patna (backend API)

FastAPI backend for the [Varuna FloodTwin](https://github.com/Maverick-Ansh/project-varuna) dashboard.
Serves the committed Patna artifact bundle and runs the differentiable flood twin live on CPU.

- Light endpoints (always on): `/api/health`, `/api/meta`, `/api/sinks`, `/api/recharge`,
  `/api/alerts`, `/api/validation`, `/api/canal_plan`, `/api/image/{name}`
- Live endpoints (CPU torch): `POST /api/whatif`, `/api/canals`, `/api/optimize`
- `POST /api/chat` — optional, uses a free hosted LLM if `LLM_API_KEY` is set (else 503)

Honest validation: dynamic twin vs Sentinel-1 SAR mean CSI ≈ 0.033; canal optimizer cuts 20.2% of
built-land flood volume; adaptive storage sizes 727/1359/2063 sites for 30/50/70% cut. See the repo.

This Space is built from the repo's `Dockerfile`. Set `LLM_API_KEY` (and optionally `LLM_API_BASE`,
`LLM_MODEL`) as Space secrets to enable chat.
