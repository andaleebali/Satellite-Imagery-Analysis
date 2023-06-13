import os
from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff

out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")

sam_kwargs = {
    "points_per_side": 100,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "min_mask_region_area": 100
}

sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    sam_kwargs=sam_kwargs,
    )

sam.generate('NDVI_Subset3_8_bit_export.tif', output="masks.tif", foreground=True, unique=True)
sam.show_masks(cmap="binary_r")

sam.tiff_to_vector("masks.tif", "masks_100.shp")