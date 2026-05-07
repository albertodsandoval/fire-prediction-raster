from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from pipeline.grid import TARGET_CRS


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
    day_df = dataset.loc[dataset["date"] == export_date].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for date {export_date}.")
    if variable not in day_df.columns:
        raise KeyError(f"Variable '{variable}' was not found in the dataset.")

    raster = np.full((height, width), nodata, dtype=np.float32)
    raster[day_df["row"].to_numpy(), day_df["col"].to_numpy()
           ] = day_df[variable].to_numpy(dtype=np.float32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

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
    x_origin = transform.c
    y_origin = transform.f

    geometries = [
        box(
            x_origin + (col * cell_size),
            y_origin - ((row + 1) * cell_size),
            x_origin + ((col + 1) * cell_size),
            y_origin - (row * cell_size),
        )
        for row, col in zip(cell_lookup["row"], cell_lookup["col"])
    ]

    grid_gdf = gpd.GeoDataFrame(
        cell_lookup.copy(), geometry=geometries, crs=TARGET_CRS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid_gdf.to_file(output, layer=layer_name, driver="GPKG")
    return output
