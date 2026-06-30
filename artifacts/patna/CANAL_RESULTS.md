# Phase C — Canal / drainage simulations on real Patna data

Differentiable flood twin (128x128 @ 60 m), 100 mm design storm, flooded volume on built land.

## Headline
| strategy | flooded volume (m3) | reduction |
|---|---|---|
| do nothing | 1,408,075 | - |
| storage pits only | 1,297,672 | 7.8% |
| **canals + pits (v2)** | **1,104,464** | **20.2%** |

v2 canal router (basin sources, downhill-guaranteed Dijkstra, descending carved bed, river/JRC
outfalls) cuts **20.2%** of built-land flood volume vs the v1 least-cost router's 16.1%.
Gradient depth-tuning (Adam through the sim) did not improve on fixed 2 m depth here
(20.2% -> 20.2%): the 2 basins are already drained; residual flood is co-located
elsewhere — consistent with the SAR-calibration finding that depth/roughness move water level, not
location.

## Dose-response (design curve)
- 25 mm -> 48,110 m3 (1,094,400 m2, peak 0.993 m)
- 50 mm -> 375,845 m3 (4,410,000 m2, peak 2.263 m)
- 75 mm -> 868,161 m3 (7,585,200 m2, peak 2.402 m)
- 100 mm -> 1,438,983 m3 (9,990,000 m2, peak 2.461 m)
- 150 mm -> 2,835,110 m3 (13,608,000 m2, peak 4.604 m)
- 200 mm -> 4,494,266 m3 (16,344,000 m2, peak 5.356 m)

## Canals
- canal 0: 2700 m, drains basin 359,407 m3, [25.57558, 85.14466] -> [25.59929, 85.15274]
- canal 1: 480 m, drains basin 196,581 m3, [25.59552, 85.15328] -> [25.59929, 85.15274]

## Figures (artifacts/patna/figures/)
- `canal_plan.png` — flood before/after with canal routes (red) + storage pits
- `dose_response.png` — flooded volume vs rainfall 25-200 mm
- `strategy_sweep.png` — do-nothing vs pits vs canals+pits
- `mass_curve.png` — domain water storage rise & drain over the storm
- `flood_timelapse.gif` — depth evolution (37 frames)
