import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin, xy
from shapely.geometry import box

TARGET_CRS = "EPSG:3310"
GRIDMET_BOUNDS = (-124.7666, 25.0666, -67.0583, 49.4000)


def _read_shapefile(path: str) -> gpd.GeoDataFrame:
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    return gpd.read_file(path)


def load_region(region_mode: str) -> gpd.GeoDataFrame:
    if region_mode == "california":
        states = _read_shapefile("assests\\CA_State.shp")
        for column, value in (("NAME", "California"), ("STUSPS", "CA"), ("STATEFP", "06")):
            if column in states.columns:
                region = states.loc[states[column] == value].copy()
                if not region.empty:
                    return region.to_crs(TARGET_CRS)
        raise ValueError("Could not find a California feature in assests\\CA_State.shp.")

    if region_mode == "sequoia":
        parks = _read_shapefile("assests\\nps_boundary.shp")
        if "UNIT_CODE" not in parks.columns:
            raise ValueError("Expected UNIT_CODE in assests\\nps_boundary.shp.")
        region = parks.loc[parks["UNIT_CODE"] == "SEQU"].copy()
        if region.empty:
            raise ValueError("Could not find UNIT_CODE='SEQU' in assests\\nps_boundary.shp.")
        return region.to_crs(TARGET_CRS)

    raise ValueError("REGION_MODE must be either 'california' or 'sequoia'.")


def build_target_grid(region: gpd.GeoDataFrame, cell_size: int) -> tuple[object, int, int, np.ndarray]:
    minx, miny, maxx, maxy = region.total_bounds
    width = int(np.ceil((maxx - minx) / cell_size))
    height = int(np.ceil((maxy - miny) / cell_size))
    transform = from_origin(minx, maxy, cell_size, cell_size)

    mask = rasterize(
        [(region.geometry.union_all(), 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    return transform, width, height, mask


def build_cell_lookup(mask: np.ndarray, transform) -> pd.DataFrame:
    rows, cols = np.where(mask)
    xs, ys = xy(transform, rows, cols, offset="center")
    return pd.DataFrame(
        {
            "cell_id": np.arange(rows.size),
            "row": rows,
            "col": cols,
            "cx": np.asarray(xs),
            "cy": np.asarray(ys),
        }
    )


def build_request_tiles(region: gpd.GeoDataFrame, tile_size_deg: float, padding_deg: float = 0.0) -> list[object]:
    region_wgs84 = region.to_crs(4326)
    geom = region_wgs84.geometry.union_all()
    west, south, east, north = GRIDMET_BOUNDS
    minx, miny, maxx, maxy = geom.bounds
    minx = max(minx, west)
    miny = max(miny, south)
    maxx = min(maxx, east)
    maxy = min(maxy, north)

    tiles = []
    x_edges = np.arange(minx, maxx, tile_size_deg)
    y_edges = np.arange(miny, maxy, tile_size_deg)

    for x0 in x_edges:
        for y0 in y_edges:
            x1 = min(x0 + tile_size_deg, maxx)
            y1 = min(y0 + tile_size_deg, maxy)
            tile = box(x0, y0, x1, y1)
            if not geom.intersection(tile).is_empty:
                req_west = max(x0 - padding_deg, west)
                req_south = max(y0 - padding_deg, south)
                req_east = min(x1 + padding_deg, east)
                req_north = min(y1 + padding_deg, north)
                tiles.append(
                    box(req_west, req_south, req_east, req_north)
                )

    return tiles
