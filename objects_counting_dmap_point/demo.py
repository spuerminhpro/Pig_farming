import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from Pig_farming.objects_counting_dmap.model_ import UNet, FCRN_A
import json
from datetime import datetime

def create_demo_dataset(input_dir, output_dir, model_path, model_type='UNet', ground_truth=None):
    """
    Create a demo dataset with input images and their density maps.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        model_path: Path to the trained model
        model_type: Type of model to use ('UNet' or 'FCRN_A')
        ground_truth: Optional dictionary mapping image names to true counts
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'input_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'density_maps'), exist_ok=True)
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters should match those used in training
    if model_type == 'UNet':
        # UNet parameters from train.py (unet_filters=64, convolutions=2)
        model = UNet(input_filters=3, filters=64, N=2)
    else:
        model = FCRN_A(input_filters=3, N=2)
    
    # Load model weights
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Store all results for analysis
    results = {
        'model_type': model_type,
        'model_path': model_path,
        'predicted_counts': [],
        'image_names': [],
        'errors': [] if ground_truth else None,
    }
    
    # Process each image
    total_processed = 0
    total_error = 0.0
    
    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Load and preprocess image
        img_path = os.path.join(input_dir, img_name)
        original_img = Image.open(img_path).convert('RGB')
        
        # Convert to tensor as required by the model
        img_tensor = TF.to_tensor(original_img).unsqueeze(0).to(device)
        
        # Generate density map with torch.no_grad() to save memory
        with torch.no_grad():
            density_map = model(img_tensor)
            
        # Calculate count (normalized by 100 as in looper.py)
        count = torch.sum(density_map).item() / 100.0
        
        # Calculate error if ground truth is provided
        error = None
        if ground_truth and img_name in ground_truth:
            true_count = ground_truth[img_name]
            error = abs(true_count - count)
            total_error += error
            results['errors'].append(error)
            print(f"True count: {true_count:.2f}, Predicted count: {count:.2f}, Error: {error:.2f}")
        else:
            print(f"Predicted count: {count:.2f}")
        
        results['predicted_counts'].append(count)
        results['image_names'].append(img_name)
        
        # Convert density map to numpy array
        density_map_np = density_map.squeeze().cpu().detach().numpy()
        
        # Save input image
        input_save_path = os.path.join(output_dir, 'input_images', img_name)
        original_img.save(input_save_path)
        
        # Enhance visualization of low-density areas
        epsilon = 1e-6
        # Use logarithmic scaling to make features more visible
        log_density = np.log(density_map_np + epsilon)
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title('Input Image')
        plt.axis('off')
        
        # Density map
        plt.subplot(1, 3, 2)
        im = plt.imshow(density_map_np, cmap='hot', interpolation='bilinear')
        plt.colorbar(im)
        plt.title(f'Density Map')
        plt.axis('off')
        
        # Overlay with enhanced visualization
        plt.subplot(1, 3, 3)
        plt.imshow(original_img)
        overlay = plt.imshow(log_density, cmap='jet', alpha=0.7)
        plt.colorbar(overlay)
        plt.title(f'Overlay (Count: {count:.2f})')
        plt.axis('off')
        
        # Save visualization
        density_save_path = os.path.join(output_dir, 'density_maps', 
                                       f'density_{os.path.splitext(img_name)[0]}.png')
        plt.savefig(density_save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        total_processed += 1
    
    # Calculate summary statistics
    avg_count = np.mean(results['predicted_counts'])
    median_count = np.median(results['predicted_counts'])
    min_count = np.min(results['predicted_counts'])
    max_count = np.max(results['predicted_counts'])
    
    # Calculate mean error if ground truth was provided
    mean_error = total_error / total_processed if ground_truth else None
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = {
        'model_type': model_type,
        'model_path': model_path,
        'total_images': total_processed,
        'statistics': {
            'average_count': float(avg_count),
            'median_count': float(median_count),
            'min_count': float(min_count),
            'max_count': float(max_count),
            'mean_error': float(mean_error) if mean_error is not None else None,
        },
        'per_image': [
            {
                'image': img, 
                'predicted_count': float(count),
                'error': float(err) if err is not None else None
            }
            for img, count, err in zip(
                results['image_names'], 
                results['predicted_counts'],
                results['errors'] if results['errors'] else [None] * len(results['image_names'])
            )
        ]
    }
    
    # Save statistics to JSON file
    json_path = os.path.join(results_dir, f'demo_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create summary visualization
    plt.figure(figsize=(15, 10))
    
    # Plot histogram of counts
    plt.subplot(2, 2, 1)
    plt.hist(results['predicted_counts'], bins=min(20, total_processed), color='skyblue', edgecolor='black')
    plt.axvline(avg_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_count:.2f}')
    plt.axvline(median_count, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_count:.2f}')
    plt.title('Distribution of Counts')
    plt.xlabel('Count')
    plt.ylabel('Number of Images')
    plt.legend()
    
    # Plot counts by image
    display_limit = min(20, total_processed)
    plt.subplot(2, 2, 2)
    indices = np.arange(display_limit)
    truncated_names = [name[:10] + '...' if len(name) > 13 else name 
                     for name in results['image_names'][:display_limit]]
    plt.bar(indices, results['predicted_counts'][:display_limit], color='skyblue')
    plt.xticks(indices, truncated_names, rotation=45, ha='right')
    plt.title('Counts by Image')
    plt.xlabel('Image')
    plt.ylabel('Count')
    
    # Add error plot if ground truth was provided
    if ground_truth:
        plt.subplot(2, 2, 3)
        plt.bar(indices, results['errors'][:display_limit], color='salmon')
        plt.xticks(indices, truncated_names, rotation=45, ha='right')
        plt.title(f'Errors by Image (Mean: {mean_error:.2f})')
        plt.xlabel('Image')
        plt.ylabel('Absolute Error')
    
    plt.tight_layout()
    summary_path = os.path.join(results_dir, f'summary_{timestamp}.png')
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\nProcessing complete! {total_processed} images processed.")
    print(f"Results saved to {output_dir}")
    print(f"Average count: {avg_count:.2f}")
    if mean_error is not None:
        print(f"Average error: {mean_error:.2f}")

if __name__ == "__main__":
    # Example usage
    input_dir = "/mnt/data/PythonProject/Pig_counting/dataset/output_dataset_test/valid/images"  # Directory containing input images
    output_dir = "/mnt/data/PythonProject/Pig_counting/Pig_farming/objects_counting_dmap/demo_results"  # Directory to save results
    model_path = "/mnt/data/PythonProject/Pig_counting/Pig_farming/objects_counting_dmap/checkpoints/pig_farming_UNet_epoch23.pth"  # Path to trained model
    model_type = "UNet"  # or "FCRN_A"
    
    # Optional: If you have ground truth counts, provide them as a dictionary
    # ground_truth = {
    #     "image1.jpg": 8,
    #     "image2.jpg": 12,
    #     # etc.
    # }
    
    create_demo_dataset(input_dir, output_dir, model_path, model_type)  # Add ground_truth=ground_truth if available