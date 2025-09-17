from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare models
## Grounding SAM
object_detector, sam_processor, sam_segmentator = prepare_model(
    device=DEVICE,
    detector_id="IDEA-Research/grounding-dino-base",
    segmenter_id="facebook/sam-vit-base",
)