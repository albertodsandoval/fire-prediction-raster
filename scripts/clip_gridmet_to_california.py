from pathlib import Path
import sys

import numpy as np
import rioxarray  # noqa: F401 - registers the .rio accessor on xarray objects
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.grid import load_region

INPUT_DIR = Path(r"D:\Development\Datasets\Sequoia\gridMET netCDF")
OUTPUT_DIR = Path(r"D:\Development\Datasets\Sequoia\gridMET netCDF\california")
VARIABLES = ["tmmx", "tmmn", "pr", "vs", "rmax"]
YEARS = [2020]
CLIP_TO_SHAPE = True


def open_gridmet_variable(variable: str, year: int, input_dir: Path) -> xr.DataArray:
    path = input_dir / f"{variable}_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing input NetCDF: {path}")

    ds = xr.open_dataset(path)
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
            raise KeyError(f"Could not resolve '{variable}' in {path}.")
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


def clip_variable_to_california(variable: str, year: int, input_dir: Path, output_dir: Path) -> Path:
    california = load_region("california").to_crs(4326)
    minx, miny, maxx, maxy = california.total_bounds

    da = open_gridmet_variable(variable, year, input_dir)
    lat_descending = da["lat"][0] > da["lat"][-1]
    if bool(lat_descending):
        subset = da.sel(lat=slice(maxy, miny), lon=slice(minx, maxx))
    else:
        subset = da.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))

    subset = subset.astype(np.float32)
    subset = subset.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs(4326)
    subset.name = variable
    subset.attrs["source_variable"] = variable
    subset.attrs["subset_region"] = "California"

    if CLIP_TO_SHAPE:
        subset = subset.rio.clip(california.geometry, california.crs, drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variable}_{year}_california.nc"
    subset.to_dataset(name=variable).to_netcdf(output_path)
    return output_path


def main() -> None:
    for variable in VARIABLES:
        for year in YEARS:
            output = clip_variable_to_california(variable, year, INPUT_DIR, OUTPUT_DIR)
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
