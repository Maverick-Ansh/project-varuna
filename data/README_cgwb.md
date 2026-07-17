# CGWB Dynamic Ground Water Resources 2024 — Karnataka

`cgwb_karnataka_2024_district.csv` — district-wise groundwater availability, utilization and
extraction for Karnataka, assessment year 2024. Quantities are in **Ham (hectare-metres)**;
`Stage of GW extraction (%)` = total annual extraction / annual extractable resource.

## Provenance

- Original source: Central Ground Water Board (CGWB), *National Compilation on Dynamic Ground
  Water Resources of India, 2024*, Ministry of Jal Shakti.
- Machine-readable mirror used here: OpenCity CKAN dataset
  "National Compilation on Dynamic Ground Water Resources of India 2024"
  (<https://data.opencity.in/dataset/national-compilation-on-dynamic-ground-water-resources-of-india-2024>),
  resource "Karnataka Groundwater Availability Utilization and Extraction 2024".
  Downloaded 2026-07-17. 31 districts + 2 `Total` rows (dropped by the loader).
- The categorization column (Safe / Semi-critical / Critical / Over-exploited) is **derived**
  by `varuna.build.state_screen.category_from_stage` using the GEC-2015 stage bands
  (>100 / 90–100 / 70–90 / ≤70 %). The published CGWB category also validates against
  water-level trends; for Karnataka 2024 the stage bands reproduce the well-known units
  (Kolar 193%, Bengaluru Urban 187%, Chikkaballapur 164% — all Over-exploited).

## Resolution honesty

This file is **district-level** (`admin_level: "district"` in every output built from it).
CGWB assesses Karnataka per **taluk**; the taluk table exists only inside the state-report
PDF annexures / India-WRIS GEC portal. Upgrade path (plan §1.1):

1. Get taluk polygons (datameet/maps 2011 frame or KGIS) → upload as an EE asset.
2. Parse the taluk annexure of the CGWB Karnataka 2024 state PDF → `cgwb_karnataka_2024_taluk.csv`.
3. `python scripts/spike_boundaries.py --asset <ee-asset>` → join coverage verdict.
4. Only then run the screen with `--admin-level taluk`.

Never present a district product with taluk-sounding precision.
