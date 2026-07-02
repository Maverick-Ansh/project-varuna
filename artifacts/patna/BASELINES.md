# Baseline skill comparison vs Sentinel-1 SAR (8 storms, 2023-2025)

Single global threshold per method (best mean CSI); same domain grid + permanent-water mask for all.

| method | best threshold | mean CSI |
|---|---|---|
| static depression depth | 0.6 | **0.032** |
| TWI (topographic wetness) | 13.4526 | **0.042** |
| HAND-lite (nearest drainage) | 8.5705 | **0.051** |
| dynamic twin (w=5d) | 0.02 | **0.041** |

Per-date CSI in baseline_comparison.json. Antecedent-window sweep for the twin included.