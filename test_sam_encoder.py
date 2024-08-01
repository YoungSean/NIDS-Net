
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
sam = SegmentAnythingPredictor(vit_model="vit_h")

# sam_input = sam.preprocess(rgb_imgs)
# z_rgb = sam.image_encoder(sam_input)