# This script runs the automatic mask generator from segment-geospatial
import os
from samgeo import SamGeo

# Downloads the pre-trained segment anything model if it is not available
out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")

#Defines the parameters for the model that are not default
sam_kwargs = {
    "points_per_side": 100,
    "crop_n_layers": 1,
    "min_mask_region_area": 5,
    "pred_iou_thresh":0.86,
    "stability_score_thresh": 0.75,
}


# Starts the model
sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    sam_kwargs=sam_kwargs
    )

#Generates the mask
sam.generate(
    'Subset3_1.tif', 
    output="mask.tif", 
    foreground=True, 
    unique=True
    )

# Displays a visualisation of the mask
sam.show_masks(cmap="binary_r")

#Converts the tif to a shapefile format
sam.tiff_to_vector("mask.tif", "mask.shp")