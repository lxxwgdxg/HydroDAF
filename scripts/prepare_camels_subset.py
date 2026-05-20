import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


DEFAULT_SELECTION = {
    "huc01": {
        "source": "01013500",
        "targets": ["01022500", "01144000", "01169000"],
    },
    "huc03": {
        "source": "02472000",
        "targets": ["02143040", "02198100", "02469800"],
    },
}


def read_attr_tables(attr_dir: Path) -> pd.DataFrame:
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(attr_dir / name, sep=";", dtype={"gauge_id": str})

    names = read("camels_name.txt")
    topo = read("camels_topo.txt")
    clim = read("camels_clim.txt")
    hydro = read("camels_hydro.txt")
    return names.merge(topo, on="gauge_id").merge(clim, on="gauge_id").merge(hydro, on="gauge_id")


def huc_for(attrs: pd.DataFrame, gauge_id: str) -> str:
    row = attrs[attrs["gauge_id"] == gauge_id]
    if row.empty:
        raise ValueError(f"Gauge {gauge_id} not found in CAMELS attributes")
    return str(row.iloc[0]["huc_02"]).zfill(2)


def read_streamflow(z: ZipFile, gauge_id: str, huc: str) -> pd.DataFrame:
    name = f"basin_dataset_public_v1p2/usgs_streamflow/{huc}/{gauge_id}_streamflow_qc.txt"
    with z.open(name) as fh:
        df = pd.read_csv(
            fh,
            delim_whitespace=True,
            header=None,
            names=["gauge_id", "Year", "month", "day", "Data", "qc"],
            dtype={"gauge_id": str},
        )
    df["date"] = pd.to_datetime(df[["Year", "month", "day"]])
    return df[["date", "Year", "month", "day", "Data"]]


def read_forcing_file(z: ZipFile, name: str) -> pd.DataFrame:
    with z.open(name) as fh:
        lines = fh.read().decode("utf-8", errors="replace").splitlines()
    data = "\n".join(lines[4:])
    df = pd.read_csv(
        BytesIO(data.encode("utf-8")),
        delim_whitespace=True,
        header=None,
        names=["Year", "month", "day", "hour", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"],
    )
    df["date"] = pd.to_datetime(df[["Year", "month", "day"]])
    return df[["date", "prcp"]]


def read_daymet_prcp(z: ZipFile, gauge_id: str, huc: str) -> pd.DataFrame:
    base = f"basin_dataset_public_v1p2/elev_bands_forcing/daymet/{huc}"
    list_name = f"{base}/{gauge_id}.list"
    with z.open(list_name) as fh:
        lines = fh.read().decode("utf-8", errors="replace").splitlines()
    entries = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            entries.append((parts[0], float(parts[1])))
    if not entries:
        raise ValueError(f"No elevation-band forcing entries for {gauge_id}")

    total_weight = sum(w for _, w in entries)
    acc = None
    for filename, weight in entries:
        forcing_name = f"{base}/{filename}"
        fdf = read_forcing_file(z, forcing_name)
        weighted = fdf["prcp"].values * (weight / total_weight)
        if acc is None:
            acc = pd.DataFrame({"date": fdf["date"], "P_daymet": weighted})
        else:
            acc["P_daymet"] += weighted
    return acc


def extract_gauge(z: ZipFile, attrs: pd.DataFrame, gauge_id: str) -> pd.DataFrame:
    huc = huc_for(attrs, gauge_id)
    q = read_streamflow(z, gauge_id, huc)
    p = read_daymet_prcp(z, gauge_id, huc)
    df = q.merge(p, on="date", how="inner")
    df = df[df["Data"] > 0].copy()
    return df[["Year", "month", "day", "Data", "P_daymet"]]


def write_group(group_name: str, gauges: list[str], z: ZipFile, attrs: pd.DataFrame, out_root: Path) -> None:
    out_dir = out_root / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    area_rows = []
    for gauge_id in gauges:
        print(f"[extract] {group_name} {gauge_id}")
        df = extract_gauge(z, attrs, gauge_id)
        df.to_excel(out_dir / f"{gauge_id}.xlsx", index=False)
        a = attrs[attrs["gauge_id"] == gauge_id].iloc[0].to_dict()
        meta_rows.append(a)
        area_rows.append({"basin": gauge_id, "area_km2": float(a["area_gages2"])})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / "selected_station_metadata.csv", index=False)
    pd.DataFrame(area_rows).to_csv(out_dir / "basin_areas.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/camels_us/raw/basin_timeseries_v1p2_metForcing_obsFlow.zip")
    parser.add_argument("--attr-dir", default="data/camels_us/attributes")
    parser.add_argument("--out-root", default="data/camels_us/selected")
    args = parser.parse_args()

    attrs = read_attr_tables(Path(args.attr_dir))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.zip) as z:
        for group_name, spec in DEFAULT_SELECTION.items():
            gauges = [spec["source"], *spec["targets"]]
            write_group(group_name, gauges, z, attrs, out_root)

    selection_rows = []
    for group_name, spec in DEFAULT_SELECTION.items():
        for role, gauges in [("source", [spec["source"]]), ("target", spec["targets"])]:
            for gauge_id in gauges:
                selection_rows.append({"group": group_name, "role": role, "gauge_id": gauge_id})
    pd.DataFrame(selection_rows).to_csv(out_root / "benchmark_selection.csv", index=False)
    print(f"Saved CAMELS subset to {out_root}")


if __name__ == "__main__":
    main()
