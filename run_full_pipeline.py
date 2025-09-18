from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare models
## Grounding SAM
object_detector, sam_processor, sam_segmentator = prepare_model(
    device=DEVICE,
    detector_id="IDEA-Research/grounding-dino-base",
    segmenter_id="facebook/sam-vit-base",
)

print(object_detector)
print(sam_processor)
print(sam_segmentator)

image_path = "/home/minhthan001/ThreeD/MIDI-3D/assets/example_data/Realistic-Style/00_rgb.png"
write_path = "/home/minhthan001/ThreeD/MIDI-3D/assets/my_results"

image = Image.open(image_path).convert("RGB")

text_labels = "a cat"
detections = detect(object_detector, image, text_labels, detect_threshold=0.3)