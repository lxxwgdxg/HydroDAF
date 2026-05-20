# HydroDAF Public CAMELS-US Benchmark

This repository is the public reproducibility package for the revised HydroDAF manuscript:

**HydroDAF: A Hydrology-Guided Domain Adaptation Framework for Streamflow Forecasting in Data-Scarce Basins**

The proprietary local case-study data used in the manuscript are not included. This package contains only the public CAMELS-US benchmark code and selected public CAMELS-US time series used to reproduce the open-data part of the revised evaluation.

## Contents

- `src/hydrodaf_benchmark.py`: HydroDAF, baseline models, training loop, and metrics.
- `scripts/run_public_benchmark.py`: runs the public CAMELS-US benchmark across the selected HUC01 and HUC03 groups.
- `scripts/prepare_camels_subset.py`: optional script for regenerating the selected public files from raw CAMELS-US downloads.
- `configs/public_benchmark_main.csv`: six-target public benchmark used for the main revised manuscript.
- `configs/public_sensitivity_02361000.csv`: supplementary sensitivity gauge where Regional LSTM is strongest.
- `data/camels_us/selected/`: processed public CAMELS-US gauge files.
- `reference_results/`: public benchmark tables reported in the revised manuscript.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS, activate with:

```bash
source .venv/bin/activate
```

## Reproduce the Public Benchmark

The command below runs Local LSTM, source-only zero-shot transfer, Fine-tuned LSTM, Regional LSTM, Shared-V DAF, and HydroDAF for the public CAMELS-US benchmark. It uses the same default settings as the revised manuscript: 30% target training data, seeds 42-44, input length 60, hidden dimension 32, and CPU execution.

```bash
python scripts/run_public_benchmark.py
```

Outputs are written to:

```text
outputs/public_camels_main/
```

The key files are:

- `public_camels_metrics_raw.csv`
- `public_camels_metrics_summary.csv`
- `public_camels_average_summary.csv`

To compare reproduced average NSE values against the manuscript reference table:

```bash
python scripts/compare_to_reference.py
```

Small numerical differences can occur across PyTorch versions and hardware backends. The reference values in `reference_results/` were generated with CPU execution.

## Run One Group Manually

HUC01:

```bash
python src/hydrodaf_benchmark.py run-group --data-dir data/camels_us/selected/huc01 --out-dir outputs/huc01 --source 01013500 --targets 01022500 01144000 01169000
```

HUC03:

```bash
python src/hydrodaf_benchmark.py run-group --data-dir data/camels_us/selected/huc03 --out-dir outputs/huc03 --source 02472000 --targets 02143040 02198100 02469800
```

Supplementary sensitivity gauge:

```bash
python src/hydrodaf_benchmark.py run-group --data-dir data/camels_us/selected/huc03 --out-dir outputs/sensitivity_02361000 --source 02472000 --targets 02361000
```

## Regenerate the Selected CAMELS-US Files

The processed gauge files are included for convenience. To regenerate them from raw public CAMELS-US downloads, place the files under:

```text
data/camels_us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip
data/camels_us/attributes/
```

Then run:

```bash
python scripts/prepare_camels_subset.py
```

The script expects the CAMELS-US attribute text files `camels_name.txt`, `camels_topo.txt`, `camels_clim.txt`, and `camels_hydro.txt`.

## Public Data Citation

The benchmark uses CAMELS-US public forcing and streamflow data. Please cite the CAMELS-US dataset and the revised HydroDAF manuscript when using this package.

## Important Privacy Note

Do not add proprietary local hydrometeorological spreadsheets to this repository. The `.gitignore` is configured to ignore arbitrary spreadsheet data and only allow the selected public CAMELS-US subset under `data/camels_us/selected/`.
