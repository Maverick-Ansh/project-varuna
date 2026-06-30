# Adaptive distributed storage — sizing to local geography

The model decides **how many** storage sites are needed and **where**, by sizing each site to its own
local depression (the depth water pools to at that gradient minimum). It is **non-linear**: deep sinks
absorb a lot per site, shallow ones little, so the flood-cut curve bends and each extra site is worth
less. All cuts are **measured by re-simulating** the 100 mm storm (twin), not assumed.

Total flood volume to manage: **1,438,983 m3** over **2,775** flooded
built cells (the addressable local minima).

## How many sites for a target cut
| target | sites needed | storage built | ~equiv 50 m3 units |
|---|---|---|---|
| 30% cut | **727** | 766,364 m3 | ~15,327 |
| 50% cut | **1,359** | 1,103,334 m3 | ~22,067 |
| 70% cut | **2,063** | 1,361,114 m3 | ~27,222 |

## Dose curve (sites -> measured flood cut)
- 50 sites -> 3.2% cut (95,046 m3 built, median site 1,736 m3)
- 100 sites -> 5.8% cut (169,980 m3 built, median site 1,601 m3)
- 200 sites -> 10.2% cut (296,548 m3 built, median site 1,376 m3)
- 500 sites -> 21.4% cut (603,011 m3 built, median site 1,130 m3)
- 1,000 sites -> 40.3% cut (962,820 m3 built, median site 872 m3)
- 2,000 sites -> 67.3% cut (1,354,224 m3 built, median site 591 m3)
- 2,775 sites -> 100.0% cut (1,438,983 m3 built, median site 421 m3)

The median site shrinks as count grows (1,736 -> 421 m3):
the model spends its first sites on the deepest sinks (biggest bang), exactly the dynamic behaviour
intended. Code: `varuna/serve/containers.py` (`plan_storage`). Figure: `figures/storage_dose.png`.
