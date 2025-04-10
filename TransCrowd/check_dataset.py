import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter

# Set dataset paths - using the same paths as in predataset_custom.py
root = '/mnt/sda1/PythonProject/Pig_counting/dataset/pigs_detection/'
output_dir = '/mnt/sda1/PythonProject/Pig_counting/dataset/TransCrowd_/'
train_images_crop = os.path.join(output_dir, 'train', 'images_crop')
train_gt_crop = os.path.join(output_dir, 'train', 'gt_density_map_crop')
test_images_crop = os.path.join(output_dir, 'val_test', 'images_crop')
test_gt_crop = os.path.join(output_dir, 'val_test', 'gt_density_map_crop')

def generate_density_map(points, shape, sigma=4):
    """Generate a density map from point annotations"""
    density_map = np.zeros(shape)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x < shape[1] and y < shape[0]:
            density_map[y, x] = 1
    density_map = gaussian_filter(density_map, sigma=sigma)
    return density_map

def visualize_images(images_crop_path, gt_crop_path, num_images=5):
    """Visualize a few images from the dataset with their density maps"""
    img_paths = glob.glob(os.path.join(images_crop_path, '*.jpg'))
    img_paths.sort()

    if not img_paths:
        print(f"No images found in {images_crop_path}")
        return

    print(f"Found {len(img_paths)} images in {images_crop_path}")
    print(f"Visualizing {min(num_images, len(img_paths))} images...")

    for idx, img_path in enumerate(img_paths[:num_images]):
        h5_path = img_path.replace('images_crop', 'gt_density_map_crop').replace('.jpg', '.h5')

        if not os.path.exists(h5_path):
            print(f"Missing density map: {h5_path}")
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load ground truth count
        with h5py.File(h5_path, 'r') as hf:
            gt_count = hf['gt_count'][()]
            
            # Check if density map is stored directly
            if 'density' in hf:
                density_map = hf['density'][()]
            # Check if points are stored in the h5 file
            elif 'points' in hf:
                points = hf['points'][()]
                # Generate density map from points
                density_map = generate_density_map(points, img.shape[:2])
            else:
                # If neither is stored, create a simple visualization
                density_map = np.zeros(img.shape[:2])
                # Add a simple visualization for density map
                density_map = np.ones(img.shape[:2]) * (gt_count / (img.shape[0] * img.shape[1]))

        print(f"Image {idx+1}: {os.path.basename(img_path)}, GT Count: {gt_count:.2f}")

        # Show image and density map side by side
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Original Image (GT Count: {gt_count:.2f})')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(density_map, cmap='jet')
        plt.title('Density Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Visualize some images from the training dataset
print("Visualizing training images...")
visualize_images(train_images_crop, train_gt_crop, num_images=3)

# Visualize some images from the test dataset
print("\nVisualizing test images...")
visualize_images(test_images_crop, test_gt_crop, num_images=2) 