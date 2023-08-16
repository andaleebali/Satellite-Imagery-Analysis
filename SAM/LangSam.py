# Script for running the LangSAM model in segment-geospatial
from samgeo.text_sam import LangSAM

sam = LangSAM() # Calls the model

text_prompt="building" # text prompt of object to be segmented

#Define the parameter for the model and run
sam.predict(
    image='M:\Documents\Dissertation\Code\Subset3_14.tif', 
    text_prompt=text_prompt, 
    box_threshold=0.1, 
    text_threshold=0.24)

# Save as a tif file
sam.show_anns(
    output='buildings.tif',
)

# Convert the saved tif file to a shapefile
sam.raster_to_vector("buildings.tif", "build.shp")
