import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from pygridmet.pygridmet import get_bygeom
from rasterstats import zonal_stats
import rioxarray
import pandas as pd

#------------ DEFINING THE REGION --------------
park = gpd.read_file("assests\\nps_boundary.shp").to_crs("EPSG:3310") # reading in boundry
sequoia = park.loc[park['UNIT_CODE']=='SEQU'].copy() # extracts only sequoia
geom = sequoia.geometry.union_all()

#--------------- CREATING GRID -----------------
# bounding box
minx, miny, maxx, maxy = sequoia.total_bounds

cell_size = 500  # meters

# generate grid
cells = []
for x in np.arange(minx, maxx, cell_size):
    for y in np.arange(miny, maxy, cell_size):
        cells.append(box(x, y, x + cell_size, y + cell_size))

grid = gpd.GeoDataFrame(geometry=cells, crs=sequoia.crs)

# clip to sequoia
grid = grid[grid.intersects(sequoia.union_all())].copy()

# assign ids + centroids
grid["cell_id"] = range(len(grid))
grid["cx"] = grid.centroid.x
grid["cy"] = grid.centroid.y


#---------- EXTRACTING GRIDMET DATA ------------
ds = get_bygeom(
    geometry=geom,
    dates=("2020-07-01", "2020-07-10"),
    crs=sequoia.crs,
    variables=["tmmx"]
)

ds = ds.rio.write_crs("EPSG:4326").rio.reproject("EPSG:3310")

def zonal_mean_for_day(day_ds, grid, path):
    day_ds = day_ds.rio.write_nodata(np.nan)
    day_ds.rio.to_raster(path)
    zs = zonal_stats(grid, path, stats=['mean'], nodata=np.nan)
    return [d['mean'] for d in zs]

rows = []
for t in ds.time.values:
    date = pd.to_datetime(t).date()

    tmmx_vals = zonal_mean_for_day(ds["tmmx"].sel(time=t), grid, f"_tmmx_{date}.tif")
    # add more vars the same way if they’re in ds:
    # pr_vals   = zonal_mean_for_day(ds["pr"].sel(time=t), grid, f"_pr_{date}.tif")

    rows.append(pd.DataFrame({
        "cell_id": grid["cell_id"].values,
        "date": date,
        "tmmx_k": tmmx_vals,
    }))
    print(rows)

df = pd.concat(rows, ignore_index=True)
print(df)