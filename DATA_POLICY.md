# Data Policy

This public package is intentionally limited to the CAMELS-US reproducibility benchmark used in the revised manuscript.

Included:

- Processed CAMELS-US public gauge files for the selected benchmark stations.
- Gauge-selection CSV files.
- Reference public benchmark metrics reported in the revised manuscript.
- Scripts for regenerating the selected public subset from CAMELS-US raw downloads.

Excluded:

- Any proprietary local case-study observations.
- Any third-party restricted hydrometeorological files.
- Any private raw spreadsheets from the authors' local study area.
- Model checkpoints generated from proprietary data.

Before publishing, run:

```bash
python scripts/run_public_benchmark.py --help
```

and inspect the repository with:

```bash
git status --ignored
```

The `.gitignore` intentionally ignores arbitrary spreadsheet files unless they are inside `data/camels_us/selected/`.
