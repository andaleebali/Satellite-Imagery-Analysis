# Classifying Buildings in High Resolution Imagery

The following code was developed to classify building rooftops in Bulawayo, Zimbabwe using WorldView-2 satellite imagery but could be used and applied to a variety of applications. More details of this can be found here. https://www.geos.ed.ac.uk/~mscgis/22-23/s1885898/

## Description
This project aims to classify building rooftops using high-resolution satellite imagery. The process involves segmenting the imagery and then classifying the segments using a Random Forest classifier.

## Prerequisites
The code uses the following Python packages:

### For `SAM/segmenting.py`:
- `os`
- `SamGeo`

### For `3_randomforest.py`:
- `sklearn`
- `numpy`
- `xml`
- `graphviz`
- `matplotlib`
- `pickle`
- `skimage`
- `cv2`
- `rasterio`

### For `4_classify_segments.py`:
- `geopandas`
- `numpy`
- `rasterio`
- `pickle`
- `matplotlib`
- `skimage`

## How to Get Started
You will require an image to be analyzed in GeoTIFF format and a shapefile of segments to be classified.

## How to Run
### Segmenting the Image
If you need to segment the image first, use the script `segmenting.py`.

### Classification
1. Update the file path to the training data and run the `randomforest.py` script. This will output a `model.pkl` file to be used for classification.
2. Update the file paths to the image and shapefile of segments and run the script `classify_segments.py`. This will output a shapefile containing the classification of each segment.

## License
Include your project's license information her
