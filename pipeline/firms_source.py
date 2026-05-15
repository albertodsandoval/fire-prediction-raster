from pathlib import Path

import pandas as pd
import geopandas as gpd
from rasterio.transform import rowcol

from pipeline.grid import TARGET_CRS


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    columns_by_lower = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        column = columns_by_lower.get(candidate.lower())
        if column is not None:
            return column
    raise KeyError(
        f"Expected one of {candidates} in FIRMS data. "
        f"Available columns: {list(df.columns)}"
    )


def _read_firms_points(firms_path: str | Path) -> gpd.GeoDataFrame:
    path = Path(firms_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find FIRMS file: {path}")

    if path.suffix.lower() == ".csv":
        firms = pd.read_csv(path)
        latitude_column = _find_column(firms, ("latitude", "lat"))
        longitude_column = _find_column(firms, ("longitude", "lon", "long"))
        return gpd.GeoDataFrame(
            firms,
            geometry=gpd.points_from_xy(
                firms[longitude_column],
                firms[latitude_column],
            ),
            crs="EPSG:4326",
        )

    firms_gdf = gpd.read_file(path)
    if firms_gdf.crs is None:
        raise ValueError(f"FIRMS file has no CRS: {path}")
    return firms_gdf


def load_firms_labels(
    firms_path: str,
    cell_lookup: pd.DataFrame,
    transform,
    width: int,
    height: int,
    dates: tuple[str, str],
    label_offset_days: int = 1,
) -> pd.DataFrame:
    firms_gdf = _read_firms_points(firms_path)
    print("RAW FIRMS:", len(firms_gdf))
    print("FIRMS SOURCE CRS:", firms_gdf.crs)
    print("FIRMS LABEL DEBUG transform:", transform)
    print("FIRMS LABEL DEBUG width/height:", width, height)

    date_column = _find_column(firms_gdf, ("acq_date", "date"))
    firms_gdf["date"] = (
        pd.to_datetime(firms_gdf[date_column]) -
        pd.Timedelta(days=label_offset_days)
    ).dt.date

    start = pd.to_datetime(dates[0]).date()
    end = pd.to_datetime(dates[1]).date()
    firms_gdf = firms_gdf[
        (firms_gdf["date"] >= start) & (firms_gdf["date"] <= end)
    ].copy()
    print("AFTER DATE FILTER:", len(firms_gdf))
    print("DATE RANGE:", firms_gdf["date"].min(), firms_gdf["date"].max())

    if firms_gdf.empty:
        return pd.DataFrame(columns=["cell_id", "date", "fire_label_t1"])

    firms_gdf = firms_gdf.to_crs(TARGET_CRS)
    print("FIRMS BOUNDS (3310):", firms_gdf.total_bounds)

    xs = firms_gdf.geometry.x.to_numpy()
    ys = firms_gdf.geometry.y.to_numpy()
    rows, cols = rowcol(transform, xs, ys)
    rows = pd.Series(rows, index=firms_gdf.index)
    cols = pd.Series(cols, index=firms_gdf.index)
    print("FIRMS ROW RANGE:", (int(rows.min()), int(rows.max())))
    print("FIRMS COL RANGE:", (int(cols.min()), int(cols.max())))
    firms_gdf["row"] = rows
    firms_gdf["col"] = cols

    firms_gdf = firms_gdf[
        (firms_gdf["row"] >= -1) &
        (firms_gdf["row"] < height) &
        (firms_gdf["col"] >= -1) &
        (firms_gdf["col"] < width)
    ].copy()
    print("AFTER GRID CLIP:", len(firms_gdf))

    lookup = cell_lookup[["cell_id", "row", "col"]].copy()

    labeled = firms_gdf.merge(
        lookup,
        on=["row", "col"],
        how="inner",
    )

    labels = (
        labeled.groupby(["cell_id", "date"])
        .size()
        .reset_index(name="fire_label_t1")
    )

    print("AFTER MERGE:", len(labeled))
    sample_columns = ["date", "row", "col", "cell_id", "geometry"]
    print("FIRMS SAMPLE ASSIGNMENTS:")
    print(labeled[sample_columns].head())

    labels["fire_label_t1"] = (labels["fire_label_t1"] > 0).astype(int)

    print(labels.head())
    print("TOTAL POSITIVE CELLS:", len(labels))

    return labels[["cell_id", "date", "fire_label_t1"]]
