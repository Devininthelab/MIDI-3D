from scripts.grounding_sam import detect, plot_segmentation, prepare_model, segment
import torch
from PIL import Image

DEVICE = "cuda"


def run_grouding_sam_text(image_path : str, text_labels : list):
    image = Image.open(image_path).convert("RGB")
    object_detector, sam_processor, sam_segmentator = prepare_model(
        device=DEVICE,
        detector_id="IDEA-Research/grounding-dino-base",
        segmenter_id="facebook/sam-vit-base",
    )

    # Run detection
    detections = detect(object_detector, image, text_labels, threshold=0.3)
    segment_kwargs = {}
    segment_kwargs["detection_results"] = detections
    # Run segmentation
    detections = segment(
            sam_processor,
            sam_segmentator,
            image,
            polygon_refinement=True,
            **segment_kwargs,
        )

    seg_map_pil = plot_segmentation(image, detections)
    save_path = image_path.replace('_rgb.png', '_seg.png')
    seg_map_pil.save(save_path)
    print(f"Saved segmentation map to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True) # Input image path
    parser.add_argument("--text_labels", type=str, nargs='+', required=True) # List of text labels to detect
    args = parser.parse_args()
    run_grouding_sam_text(args.image_path, args.text_labels)
    # python -m data_preprocessing.run_segmentation --image_path category/chair/002_rgb.png --text_labels chair