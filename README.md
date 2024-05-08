# Classifiying Buildings in High Resolution Imagery
The following code was created to classify building rooftops in Bulawayo, Zimbabwe using WorldView-2 satellite imagery. More details of this can be found here: https://www.geos.ed.ac.uk/~mscgis/22-23/s1885898/


## Prerequisites
The code uses the following Python packages:

## How to get started?
You will require an image:
to be analysed in geotiff format

Training data: 

Shapefile of segments to be classified

## How to run
If you need to segment the image first use the script segmenting.py

For classification:
Update file path to training data and run the randomforest.py script
This will output a model.pkl file to be used for classification
Update the file paths to the image and shapefile of segments and run the script classify_segments.py
This will output a shapefile containing the classification of each segment


