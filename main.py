import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from pygridmet.pygridmet import get_bygeom
from rasterstats import zonal_stats


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
help(get_bygeom)
ds = get_bygeom(
    geometry=geom,
    dates=("2020-07-01", "2020-07-10"),
    crs=sequoia.crs,
    variables=["tmmx"]
)

