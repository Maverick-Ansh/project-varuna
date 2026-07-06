# RUNBOOK — FloodGNN training run (Colab / Kaggle)

Everything below is prepared and smoke-tested end-to-end on CPU (Patna: graph 37k nodes /
80k edges, dataset 9 s, 24-epoch smoke checkpoint AUC 0.856, safe_route verified through
`/api/route`). This runbook is the one GPU session that turns it into the real thing.

**What the run produces**

| artifact | what it is |
|---|---|
| `artifacts/<area>/gnn_data.npz` | training labels per area (emulator depth sweep + drain-flow) |
| `artifacts/gnn/gnn.pt` | production checkpoint — the dashboard's route backend flips from `emulator` to `GNN` once this is on main (it ships inside the Space image) |
| `artifacts/gnn/gnn_report.json` | per-area AUC / wet-RMSE / flow-Spearman / routing quality / speed |
| `artifacts/gnn/ablation.json` | layers 0 vs 2 vs 4 — proves message passing beats a per-node MLP |
| `artifacts/gnn/transfer_no_bengaluru.json`, `transfer_no_patna.json`, `transfer_summary.json` | **the paper numbers**: zero-shot cross-city AUC |
| `artifacts/gnn/figures/` | train curve, AUC-vs-rain, pred-vs-true, route demo map |

Expected T4 timing (fp16 autocast is on by default): `--data` ~1–2 min/area,
`--train` (80 epochs × 4 areas) ~10–15 min, `--ablation` +2 trainings, `--transfer`
+2 trainings ≈ **45–60 min total**. CPU works too (~15 s/epoch for Patna alone).

## A — Colab (GPU runtime)

```python
# Cell 1 — code + deps (torch is preinstalled; rasterio is the only extra)
!git clone -b feature/gnn-path-planning https://github.com/Maverick-Ansh/project-varuna.git
%cd project-varuna
!pip -q install rasterio
```

```python
# Cell 2 — the whole suite (order matters only in that --data must precede the rest)
!python scripts/run_gnn_colab.py --data --train --ablation --transfer --eval --figures --device cuda
```

```python
# Cell 3 — eyeball the headline numbers before pushing
import json
print(json.dumps(json.load(open('artifacts/gnn/transfer_summary.json')), indent=1))
print(json.dumps(json.load(open('artifacts/gnn/ablation.json')), indent=1))
rep = json.load(open('artifacts/gnn/gnn_report.json'))
for a, v in rep['areas'].items():
    print(a, 'AUC', v['edge_auc'], 'wet-RMSE', v['wet_rmse_m'], 'routing', v['routing'].get('gnn'))
```

```python
# Cell 4 — push results (GITHUB_TOKEN must be a Colab secret; never printed)
!python scripts/run_gnn_colab.py --push --branch feature/gnn-path-planning
```

## B — Kaggle differences

* Work dir is `/kaggle/working`; Internet must be ON.
* `GITHUB_TOKEN` comes from `kaggle_secrets` — **attach the secret to THIS notebook**
  (Add-ons → Secrets), or the push fails with "No user secrets exist for kernel id".
* Same cells otherwise; 2×T4 machines just run the single-GPU code.

## C — back on the PC (after the push)

1. `git pull` on `feature/gnn-path-planning`, sanity-check `artifacts/gnn/gnn_report.json`.
2. PR → main (`gh pr create --fill --body-file <notes>`); user merges (verify `gh pr view`).
3. Redeploy the Space from main (`deploy/deploy_hf_space.py`) — the image copies
   `artifacts/`, so the checkpoint rides along and `/api/route` starts answering
   `"backend": "gnn"`. Vercel rebuilds itself on the main push.
4. Paste the transfer/ablation numbers into `paper/GNN_SECTION.md` (placeholders are
   marked `TODO-COLAB`).

## Troubleshooting

* **CUDA OOM** (unlikely: ~90k params, biggest graph 37k nodes): `--hidden 64` or `--no-amp`.
* **"no gnn_data.npz"** during `--train`: `--data` didn't run for that area — rerun
  `--data --areas <id>`.
* **Bengaluru dataset has zero flow labels**: the drain planner logged a grid fallback —
  harmless for depth training; flow metrics for that area will be null.
* **Kernel dies on import**: this stack needs no pysheds/osmnx/EE — only torch, numpy,
  scipy, rasterio, matplotlib. If something else broke the env, restart runtime and rerun
  Cell 1 only.
* **Divergent branch on push**: someone committed meanwhile — `git pull --rebase origin
  feature/gnn-path-planning` in a cell, then push again.
