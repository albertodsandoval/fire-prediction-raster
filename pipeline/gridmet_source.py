from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the .rio accessor on xarray objects
import xarray as xr
from pygridmet.pygridmet import get_bygeom
from rasterio.warp import Resampling, reproject

from pipeline.grid import TARGET_CRS


def reproject_day_to_grid(day_da, transform, width: int, height: int, resampling: Resampling) -> np.ndarray:
    dest = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=np.asarray(day_da.values, dtype=np.float32),
        destination=dest,
        src_transform=day_da.rio.transform(),
        src_crs=day_da.rio.crs,
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs=TARGET_CRS,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dest


def _build_day_frame(
    date,
    data_by_variable: dict[str, np.ndarray],
    mask: np.ndarray,
    cell_lookup: pd.DataFrame,
) -> pd.DataFrame:
    day_frame = pd.DataFrame(
        {
            "cell_id": cell_lookup["cell_id"].values,
            "date": date,
        }
    )
    for variable, dest in data_by_variable.items():
        masked = dest.copy()
        masked[~mask] = np.nan
        day_frame[variable] = masked[mask]
    return day_frame


def _load_gridmet_features_pygridmet(
    region,
    dates,
    variables,
    transform,
    width: int,
    height: int,
    mask: np.ndarray,
    cell_lookup: pd.DataFrame,
    resampling_by_variable: dict[str, Resampling],
) -> pd.DataFrame:
    geom = region.geometry.union_all()
    ds = get_bygeom(
        geometry=geom,
        dates=dates,
        crs=region.crs,
        variables=variables,
    )

    rows = []
    for t in ds.time.values:
        date = pd.to_datetime(t).date()
        data_by_variable = {}
        for variable in variables:
            day_da = ds[variable].sel(time=t).rio.write_nodata(np.nan)
            data_by_variable[variable] = reproject_day_to_grid(
                day_da,
                transform,
                width,
                height,
                resampling_by_variable.get(variable, Resampling.bilinear),
            )
        rows.append(_build_day_frame(
            date, data_by_variable, mask, cell_lookup))

    return pd.concat(rows, ignore_index=True)


def _open_local_gridmet_variable(variable: str, year: int, netcdf_dir: Path) -> xr.DataArray:
    candidate = netcdf_dir / f"{variable}_{year}.nc"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing local GridMET file: {candidate}. "
            f"Download the yearly NetCDF for this variable first."
        )

    ds = xr.open_dataset(candidate)
    if variable in ds:
        da = ds[variable]
    else:
        matched_name = None
        for data_var in ds.data_vars:
            attrs = ds[data_var].attrs
            if variable in {
                attrs.get("long_name"),
                attrs.get("standard_name"),
                attrs.get("abbr"),
            }:
                matched_name = data_var
                break

        if matched_name is None:
            available = {
                name: {
                    "long_name": ds[name].attrs.get("long_name"),
                    "standard_name": ds[name].attrs.get("standard_name"),
                }
                for name in ds.data_vars
            }
            raise KeyError(
                f"Variable '{variable}' was not found in {candidate}. "
                f"Available data variables and aliases: {available}"
            )

        da = ds[matched_name]

    rename_map = {}
    if "day" in da.dims:
        rename_map["day"] = "time"
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    return da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs(4326)


def _load_gridmet_features_local_netcdf(
    region,
    dates,
    variables,
    transform,
    width: int,
    height: int,
    mask: np.ndarray,
    cell_lookup: pd.DataFrame,
    resampling_by_variable: dict[str, Resampling],
    netcdf_dir: str,
) -> pd.DataFrame:
    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[1])
    years = range(start_date.year, end_date.year + 1)
    netcdf_path = Path(netcdf_dir)

    region_wgs84 = region.to_crs(4326)
    minx, miny, maxx, maxy = region_wgs84.total_bounds

    variables_by_date: dict[pd.Timestamp, dict[str, np.ndarray]] = {}

    for variable in variables:
        yearly_arrays = []
        for year in years:
            yearly_arrays.append(_open_local_gridmet_variable(
                variable, year, netcdf_path))

        da = xr.concat(yearly_arrays, dim="time").sortby("time")
        da = da.sel(time=slice(start_date, end_date))

        lat_descending = da["lat"][0] > da["lat"][-1]
        if bool(lat_descending):
            da = da.sel(lat=slice(maxy, miny), lon=slice(minx, maxx))
        else:
            da = da.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))

        da = da.rio.write_nodata(np.nan)

        for t in da.time.values:
            date = pd.to_datetime(t).date()
            day_da = da.sel(time=t)
            dest = reproject_day_to_grid(
                day_da,
                transform,
                width,
                height,
                resampling_by_variable.get(variable, Resampling.bilinear),
            )
            variables_by_date.setdefault(date, {})[variable] = dest

    rows = []
    for date in sorted(variables_by_date):
        rows.append(_build_day_frame(
            date, variables_by_date[date], mask, cell_lookup))

    return pd.concat(rows, ignore_index=True)


def load_gridmet_features(
    region,
    dates,
    variables,
    transform,
    width: int,
    height: int,
    mask: np.ndarray,
    cell_lookup: pd.DataFrame,
    resampling_by_variable: dict[str, Resampling],
    source_mode: str = "pygridmet",
    netcdf_dir: str = ".",
) -> pd.DataFrame:
    if source_mode == "local_netcdf":
        return _load_gridmet_features_local_netcdf(
            region=region,
            dates=dates,
            variables=variables,
            transform=transform,
            width=width,
            height=height,
            mask=mask,
            cell_lookup=cell_lookup,
            resampling_by_variable=resampling_by_variable,
            netcdf_dir=netcdf_dir,
        )

    if source_mode == "pygridmet":
        return _load_gridmet_features_pygridmet(
            region=region,
            dates=dates,
            variables=variables,
            transform=transform,
            width=width,
            height=height,
            mask=mask,
            cell_lookup=cell_lookup,
            resampling_by_variable=resampling_by_variable,
        )

    raise ValueError(
        "source_mode must be either 'pygridmet' or 'local_netcdf'.")
