import geopandas as gpd
import numpy as np
from shapely.geometry import shape
import rasterio
from rasterio.features import geometry_mask
import pickle
from matplotlib import pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show

import time
start = time.time()


shppath = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/building_footprints.shp'
shapefile = gpd.read_file(filename=shppath)

print(shapefile.crs)

tiffile = '/home/s1885898/scratch/data/Subset3_nirrg.tif'
opentif = rasterio.open(fp=tiffile)

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
        geometry_list = [geometry]  # Create a list containing the single geometry
        masked_img, out_transform = mask(shapes=geometry_list, dataset=opentif, crop=True)
        clipped_tiff = opentif.read(masked=True)[0]
        #show(masked_img, 3)
        #plt.show()
        normalise_tiff = masked_img / 255
        resized_tiff = np.resize(masked_img, (1, 7500))  # Resize to 10,000 features
        #plt.imshow(resized_tiff)
        #plt.show()

        predictions = loaded_model.predict(resized_tiff)
        prediction = predictions[0]
        shapefile.at[index, 'predictions'] = prediction  # Assign the prediction to the new attribute
        print(count, prediction)
        count += 1
    except Exception as e:
        print(f"Error processing geometry at index {index}: {e}")
        continue  # Skip to the next iteration if an error occurs

# Save the updated shapefile
output_shapefile_path = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/all_buildings_predictions.shp'
shapefile.to_file(output_shapefile_path)
opentif.close()

end = time.time()
totaltime = end - start
print ("\n" + str(totaltime))