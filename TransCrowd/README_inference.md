# TransCrowd Pig Counting Inference

This guide explains how to use the inference script for the TransCrowd model adapted for pig counting.

## Prerequisites

Before using the inference script, make sure you have the following dependencies installed:

```bash
pip install torch torchvision pillow numpy opencv-python
```

## Usage

The inference script allows you to run the pig counting model on a single image or a directory of images.

### Basic Usage

```bash
python inference.py --input path/to/image_or_directory --output path/to/results --model_path path/to/model_checkpoint
```

### Arguments

- `--input`: Path to the input image or directory containing images (required)
- `--output`: Path to save the results (default: './results')
- `--model_path`: Path to the model checkpoint (default: './save_file/Pig_detection/model_best.pth')
- `--model_type`: Type of model architecture to use: 'token' or 'gap' (default: 'token')
- `--gpu_id`: GPU ID to use (default: '0')

### Examples

1. Run inference on a single image:

```bash
python inference.py --input path/to/image.jpg
```

2. Run inference on a directory of images:

```bash
python inference.py --input path/to/image_directory --output path/to/results
```

3. Specify the model path and type:

```bash
python inference.py --input path/to/image.jpg --model_path ./Networks/model_best_A_gap.pth --model_type gap
```

## Output

For single images, the script will:
1. Display the predicted pig count
2. Save the visualization with the count overlay at the specified output path

For directories, the script will:
1. Process all images in the directory
2. Save visualizations with count overlays
3. Create a 'results.txt' file with all image names and their corresponding counts

## Troubleshooting

If you encounter issues with the script:

1. Make sure you have all required dependencies installed
2. Verify that the model checkpoint exists at the specified path
3. Check that the input images are readable by PIL (supports JPG, PNG, etc.)
4. For large images, ensure you have enough GPU memory, or use CPU by setting the environment variable: `CUDA_VISIBLE_DEVICES=""`

## Model Types

- `token`: Uses the VisionTransformer_token model architecture
- `gap`: Uses the VisionTransformer_gap model architecture 