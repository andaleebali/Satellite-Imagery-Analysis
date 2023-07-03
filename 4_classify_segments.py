import geopandas as gpd
import numpy as np
import rasterio
import pickle
from matplotlib import pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show
import time
from skimage.transform import resize

start = time.time()

def prepare_roof(geometry, opentif):
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

        #rgb[:, :, 0] = red_masked
        #rgb[:, :, 1] = green_masked
        #rgb[:, :, 2] = blue_masked

    #plt.imshow(rgb)
    #plt.show()

    # Normalize the RGB values
    max_red = np.max(red_masked)
    max_green = np.max(green_masked)
    max_blue = np.max(blue_masked)

    normalized_red = red_masked / max_red
    normalized_green = green_masked / max_green
    normalized_blue = blue_masked / max_blue

    normalized_tiff = np.dstack((red_masked, green_masked, blue_masked))

    resized_tiff = resize(rgb, (50, 50, 3))
    resized_tiff = resized_tiff.flatten()
    resized_tiff = resized_tiff.reshape(1, -1)
    return resized_tiff

tiffile = '/home/s1885898/Documents/Dissertation/Code/Subset3_rgb.tif'
opentif = rasterio.open(fp=tiffile)

shppath = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/building_footprints.shp'
shapefile = gpd.read_file(filename=shppath)

print(shapefile.crs)

shapefile = shapefile.to_crs(opentif.crs)

modelpath = 'model1.pkl'
loaded_model = pickle.load(open(modelpath, "rb"))

# Create a new attribute column for predictions
shapefile['predictions'] = None
count=0

for index, row in shapefile.iterrows():
    try:
        geometry = row.geometry
        if not geometry.is_valid:
            continue  # Skip invalid geometries
        print(geometry)
        resized_tiff=prepare_roof(geometry, opentif)

        predictions = loaded_model.predict(resized_tiff)
        prediction = predictions[0]
        shapefile.at[index, 'predictions'] = prediction  # Assign the prediction to the new attribute
        print(count, prediction)
        count += 1

    except Exception as e:
        print(f"Error processing geometry at index {index}: {e}")
        continue  # Skip to the next iteration if an error occurs

# Save the updated shapefile
output_shapefile_path = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/all_buildings_predictions_rgb_remblack.shp'
shapefile.to_file(output_shapefile_path)
opentif.close()

end = time.time()
totaltime = end - start
print ("\n" + str(totaltime))