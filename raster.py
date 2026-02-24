import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from pygridmet.pygridmet import get_bygeom


#------------ DEFINING THE REGION --------------
park = gpd.read_file("assests\\nps_boundary.shp")
park = park.to_crs("EPSG:3310")
geom = park.geometry.union_all()

#--------------- CREATING GRID -----------------
# bounding box
minx, miny, maxx, maxy = park.total_bounds

cell_size = 500  # meters

# generate grid
cells = []
for x in np.arange(minx, maxx, cell_size):
    for y in np.arange(miny, maxy, cell_size):
        cells.append(box(x, y, x + cell_size, y + cell_size))

grid = gpd.GeoDataFrame(geometry=cells, crs=park.crs)

# clip to park
grid = grid[grid.intersects(park.union_all())].copy()

# assign ids + centroids
grid["cell_id"] = range(len(grid))
grid["cx"] = grid.centroid.x
grid["cy"] = grid.centroid.y


#---------- EXTRACTING GRIDMET DATA ------------
help(get_bygeom)
ds = get_bygeom(
    geometry=geom,
    dates=("2020-07-01", "2020-07-10"),
    crs=park.crs,
    variables=["tmmx"],
    conn_timeout=3000,          # seconds
    validate_filesize=False     # reduces retries if server returns partial
)