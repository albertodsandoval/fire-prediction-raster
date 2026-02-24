# Data Science Capstone Project
This repository serves to hold documentation and source code for my senior design project as a Data Science minor. I will also explain the project, the plan, and the expected outcomes in this very document. Feel free to reach me at alberto.sandoval.domingo@gmail.com for any questions/issues.

## The Idea
In our project class, we had full creative control over what we wanted to do for our capstone project. I have had an interest in Geospatial data for a while now, but have yet to get hands on with it. I took this opportunity to learn more.

### Fire Prediction Within Sequoia National Park
I am from the central valley of California, which means Sequoia is nearby. Myself and my family have always enjoyed visiting and throughout the years I have become more and more interested in hiking/backpacking. Initially, I wanted this project to be more geared towards backpacking suitability prediction, but after considering what makes a trail suitable and how difficult that data would be to capture with consistency, I pivoted for fire risk prediction.

The ultimate goal is:
	Be able to predict to probability of a fire occuring 1 day into the future, based on todays features and lagged features as well (meaning information from past days, in my case -1 and -7 days).

### Data
As for the data I will be using, I plan on leveraging:
* gridMET (Weather Data)
* FIRMS (Fire Points)
* LANDFIRE (Fuels)

From these, the Fire Points would be my labels and I would extract predictive features from gridMET and LANDFIRE, notably:
* Daily Max Temp (tmmx)
* Daily Min Temp (tmmn)
* Precipitation 
* Relative Humidity (Max)
* Relative Humidity (Min)
* Wind Speed
* Vapor Pressure Deficit
* Fuel Model
* Canopy Cover
* Canopy Height
* Canopy Base Height
* Canopy Bulk Density
* Vegetation
* FIRMS (1 if fire was detected, label)
**Note:** These features are subject to change.

## The Process
### Step 1) Defining the Region
Find a vector file for outlining the region I would like to focus on (.shp file for Sequoia)
### Step 2) Building the Grid
Create a grid within the vector Shape file for Sequoia for the sake of building a new raster with layers from other rasters
### Step 3) Load in Rasters
Load in all the data from gridMET, LANDFIRE, and FIRMS. Collect data from the same time period and ensure they all contain the entire region of Sequoia (outline by the .shp file).
### Step 4) Zonal Statistics (Reprojection)
Project data from the rasters to the same size as each cell in the grid built in step 2. We use zonal statistics for this, available with rasterio. At this step, we also must make sure the CRS aligns for your grid and the raster you are reprojecting onto it. Reproject grid onto raster CRS before anything.
### Step 5) Modeling
Here is where we are able to actually build a model with the capacity to estimate the probability of a fire happening within a given cell. These predictions are mapped back onto each cell on the grid, meaning the can be visualized like any other raster.
### Step 6) Interpret Results
Observe which features have a larger impact on the perfomance of the model. Does that make sense? Is there anything off about the model? Does the model generally do a good job of attributing a high fire risk to areas that did get burned the following day?
### Step 7) Evaluate and Document
Evaluate the models performance and whether or not it meets the requirements we will choose for it in the near future. If it does, document results and findings to discuss with professor and class.