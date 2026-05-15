from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box

from pipeline.grid import TARGET_CRS


def cell_polygon_from_row_col(transform, row: int, col: int):
    left, bottom, right, top = window_bounds(
        Window(col_off=col, row_off=row, width=1, height=1),
        transform,
    )
    return box(left, bottom, right, top)


def export_day_variable_to_geotiff(
    dataset: pd.DataFrame,
    variable: str,
    date,
    transform,
    width: int,
    height: int,
    output_path: str,
    nodata: float = np.nan,
) -> Path:
    export_date = pd.to_datetime(date).date()
    dataset_dates = pd.to_datetime(dataset["date"]).dt.date
    day_df = dataset.loc[dataset_dates == export_date].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for date {export_date}.")
    if variable not in day_df.columns:
        raise KeyError(f"Variable '{variable}' was not found in the dataset.")

    raster = np.full((height, width), nodata, dtype=np.float32)
    rows = day_df["row"].to_numpy(dtype=int)
    cols = day_df["col"].to_numpy(dtype=int)
    if (
        (rows < 0).any()
        or (rows >= height).any()
        or (cols < 0).any()
        or (cols >= width).any()
    ):
        raise ValueError("Dataset contains row/col values outside the raster grid.")

    raster[rows, cols] = day_df[variable].to_numpy(dtype=np.float32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        "RASTER EXPORT DEBUG:",
        {
            "variable": variable,
            "date": str(export_date),
            "transform": transform,
            "width": width,
            "height": height,
        },
    )

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=TARGET_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(raster, 1)

    return output


def export_grid_mask_to_geotiff(
    mask: np.ndarray,
    transform,
    output_path: str,
    nodata: int = 0,
) -> Path:
    raster = mask.astype(np.uint8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype="uint8",
        crs=TARGET_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(raster, 1)

    return output


def export_grid_cells_to_gpkg(
    cell_lookup: pd.DataFrame,
    transform,
    cell_size: int,
    output_path: str,
    layer_name: str = "grid_cells",
) -> Path:
    geometries = [
        cell_polygon_from_row_col(transform, row, col)
        for row, col in zip(cell_lookup["row"], cell_lookup["col"])
    ]

    print(
        "GRID VECTOR EXPORT DEBUG:",
        {
            "transform": transform,
            "cell_size": cell_size,
            "row_range": (
                int(cell_lookup["row"].min()),
                int(cell_lookup["row"].max()),
            ),
            "col_range": (
                int(cell_lookup["col"].min()),
                int(cell_lookup["col"].max()),
            ),
        },
    )
    grid_gdf = gpd.GeoDataFrame(
        cell_lookup.copy(), geometry=geometries, crs=TARGET_CRS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid_gdf.to_file(output, layer=layer_name, driver="GPKG")
    return output
