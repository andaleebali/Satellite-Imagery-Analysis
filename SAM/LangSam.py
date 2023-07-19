from samgeo import tms_to_geotiff, split_raster
from samgeo.text_sam import LangSAM

sam = LangSAM()

text_prompt="a building"

sam.predict(image='M:\Documents\Dissertation\Code\Subset3_1.tif', text_prompt=text_prompt, box_threshold=0.24, text_threshold=0.24)

sam.show_anns(
    cmap='Greys_r',
    add_boxes=False,
    alpha=1,
    title='Automatic Segmentation of Trees',
    blend=False,
    output='buildings.tif',
)
sam.raster_to_vector("buildings.tif", "buildings.shp")
#sam.tiff_to_vector("masks.tif", "masks_25.shp")


