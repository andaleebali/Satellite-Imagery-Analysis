# This script is for classifying a whole tif image with a shapefile for segmenting
import geopandas as gpd
import numpy as np
import rasterio
import pickle
from matplotlib import pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show
from skimage.transform import resize

def prepare_roof(geometry, opentif):
    """
    Preparing data so that the rest of the image is masked and the 
    classifier only sees one roof at a time. The image is also resized
    and flattened so that it is in the same format as when the data was 
    loaded into the training model.
        Parameters:
            geometry: shapefile
            opentif: raster image 
        Returns:
            resized_tiff(arr): processed image ready for classification
    """
    geometry_list = [geometry]  # Create a list containing the single geometry
    masked_img, _ = mask(shapes=geometry_list, dataset=opentif, crop=True)

    masked_img = np.transpose(masked_img, (1, 2, 0))

    red = masked_img[:, :, 0]
    green = masked_img[:, :, 1]
    blue = masked_img[:, :, 2]
    alpha = masked_img[:, :, 3]

    red_masked = red[alpha != 0]
    green_masked = green[alpha != 0]
    blue_masked = blue[alpha != 0]

    red_masked = resize(red_masked, (red.shape[0], red.shape[1]))
    green_masked = resize(green_masked, (red.shape[0], red.shape[1]))
    blue_masked = resize(blue_masked, (red.shape[0], red.shape[1]))

    rgb = np.dstack((red_masked, green_masked, blue_masked))

    #plt.imshow(rgb)
    #plt.show()

    resized_tiff = resize(rgb, (50, 50, 3))
    resized_tiff = resized_tiff.flatten()
    resized_tiff = resized_tiff.reshape(1, -1)
    return resized_tiff

# Opens the image to be classified
tiffile = '/home/s1885898/Documents/Dissertation/Code/Subset3_rgb.tif'
opentif = rasterio.open(fp=tiffile)

# Opens the shapefile containing the footprints
shppath = '/home/s1885898/Documents/Dissertation/Code/Inputs/red_building_footprints.shp'
shapefile = gpd.read_file(filename=shppath)

print(shapefile.crs)

# Converts the shapefile to the same coordinate reference system as the geotif
shapefile = shapefile.to_crs(opentif.crs)

# Loads the pre-trained model
modelpath = 'model.pkl'
loaded_model = pickle.load(open(modelpath, "rb"))

# Create a new attribute column for predictions
shapefile['predictions'] = None
count=0

# Loops through each polygon in the shapefile
for index, row in shapefile.iterrows():
    try:
        geometry = row.geometry # Checks the geometry is valid
        if not geometry.is_valid:
            continue  # Skip invalid geometries
        print(geometry)
        resized_tiff=prepare_roof(geometry, opentif) # Prepares the tiff for classification
        predictions = loaded_model.predict(resized_tiff) # The Prepares tiff is sent for classification
        prediction = predictions[0] # Extracts prediction label
        shapefile.at[index, 'predictions'] = prediction  # Assign the prediction to the new attribute
        print(count, prediction)
        count += 1

    except Exception as e:
        print(f"Error processing geometry at index {index}: {e}")
        continue  # Skip to the next iteration if an error occurs

# Save the updates to a new shapefile
output_shapefile_path = '/home/s1885898/Documents/Dissertation/Code/Outputs/building_class_pred_attempt_3.shp'
shapefile.to_file(output_shapefile_path)
opentif.close()