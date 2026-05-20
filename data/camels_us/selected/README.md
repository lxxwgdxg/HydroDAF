# Selected Public CAMELS-US Data

This folder contains only processed public CAMELS-US gauge time series used for the public benchmark in the revised HydroDAF manuscript. No proprietary local case-study data are included.

Each gauge file has the columns:

- `Year`, `month`, `day`: calendar date.
- `Data`: USGS streamflow from CAMELS-US.
- `P_daymet`: Daymet precipitation aggregated from elevation-band forcing.

The selected gauges are listed in `../../../../configs/public_benchmark_main.csv`. Gauge `02361000` is included only for the supplementary sensitivity case.
