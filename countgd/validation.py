import logging
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from app import (
    build_model_and_transforms,
    get_device,
    get_args_parser,
    generate_heatmap,
    get_xy_from_boxes,
    predict,
)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_output(image, boxes):
    """Get count, coordinates and heatmap from boxes"""
    count = len(boxes)
    xy_coords = get_xy_from_boxes(image, boxes)
    heatmap = generate_heatmap(image, boxes)
    return count, xy_coords, heatmap

def process_validation_dataset(model, transform, image_path: Path, text: str, output_dir: Path, device):
    """
    Process a single validation image and save results.
    
    Args:
        model: The trained model
        transform: Image transforms
        image_path: Path to the input image
        text: Text prompt for counting
        output_dir: Directory to save results
        device: Device to run model on
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process the image
    try:
        with Image.open(image_path) as im:
            im.load()
            original_image = im.convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

    # Run prediction
    start_time = time.time()
    boxes, _ = predict(model, transform, original_image, text, None, device)
    count, xy_coords, heatmap = get_output(original_image, boxes)
    latency = time.time() - start_time
    fps = 1 / latency if latency > 0 else float('inf')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), dpi=100)

    # Original image with ground truth points
    ax1.imshow(np.array(original_image))
    ax1.set_title("Original Image", fontsize=20)
    ax1.axis('off')

    # Prediction heatmap
    ax2.imshow(heatmap, cmap='jet')
    ax2.set_title(f"Prediction: {count} objects", fontsize=20)
    
    # Add performance metrics
    metrics_text = f"Latency: {latency:.2f} sec\nFPS: {fps:.2f}"
    ax2.text(20, 40, metrics_text, fontsize=18, color="white", backgroundcolor="black")

    # Save the results
    plt.tight_layout()
    output_plot_path = output_dir / f"{image_path.stem}_results.png"
    plt.savefig(output_plot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    logger.info(f"Processed {image_path.name}: Count={count}, Latency={latency:.2f} sec")
    return count, xy_coords, heatmap

def process_validation_folder(model, transform, input_folder: str, text: str, output_dir: str):
    """
    Process all images in a validation folder.
    
    Args:
        model: The trained model
        transform: Image transforms
        input_folder: Path to input folder containing validation images
        text: Text prompt for counting
        output_dir: Directory to save results
    """
    input_path = Path(input_folder)
    output_path = Path(output_dir)
    device = get_device()
    
    # Get all image files
    image_files = list(input_path.glob("**/*.jpg")) + list(input_path.glob("**/*.png"))
    
    for img_file in image_files:
        process_validation_dataset(
            model, transform, img_file, text, output_path, device
        )

if __name__ == "__main__":
    # Setup model
    parser = get_args_parser()
    parser.set_defaults(pretrain_model_path="checkpoint_best_regular.pth")
    args = parser.parse_args([])
    
    device = get_device()
    model, transform = build_model_and_transforms(args)
    model = model.to(device)
    
    # Process validation dataset
    process_validation_folder(
        model,
        transform,
        "path/to/validation/folder",
        "pig",
        "path/to/output/folder"
    ) 