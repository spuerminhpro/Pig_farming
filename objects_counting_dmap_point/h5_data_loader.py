import h5py
import numpy as np
import cv2
import json
import os

# Define paths
splits_path = '/mnt/data/PythonProject/Pig_counting/dataset/pig_FSC147/data_split.json'
images_dir = '/mnt/data/PythonProject/Pig_counting/dataset/pig_FSC147/images'
density_dir = '/mnt/data/PythonProject/Pig_counting/dataset/pig_FSC147/gt_density_maps'
output_dir = '/mnt/data/PythonProject/Pig_counting/Pig_farming/objects_counting_dmap/data/pig_farming'
os.makedirs(output_dir, exist_ok=True)

# Load the data splits from JSON
with open(splits_path, 'r') as f:
    splits = json.load(f)

# Fixed size for resizing (width, height)
fixed_size = (256, 256)

# Process each split: train, val, test
for split in ['train', 'val', 'test']:
    image_names = splits[split]
    images_list = []
    labels_list = []
    
    for img_name in image_names:
        # Load image using OpenCV
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}. Skipping.")
            continue
        # Convert from BGR to RGB and normalize to [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Resize image to fixed dimensions
        img = cv2.resize(img, fixed_size, interpolation=cv2.INTER_AREA)
        # Convert to channel-first: (H, W, 3) -> (3, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # Load corresponding density map
        density_name = os.path.splitext(img_name)[0] + '.npy'
        density_path = os.path.join(density_dir, density_name)
        if not os.path.exists(density_path):
            print(f"Warning: Density map {density_path} not found. Skipping.")
            continue
        density_map = np.load(density_path)
        
        # Preserve overall count during resizing
        original_sum = np.sum(density_map)
        density_map = cv2.resize(density_map, fixed_size, interpolation=cv2.INTER_LINEAR)
        if original_sum > 0:
            current_sum = np.sum(density_map)
            if current_sum > 0:
                density_map = density_map * (original_sum / current_sum)
            else:
                #skip this image
                continue
        else:
            #skip this image
            continue
        
        # Expand dimensions to match expected (1, H, W) format
        density_map = np.expand_dims(density_map, axis=0)
        
        # Append processed data to lists
        images_list.append(img)
        labels_list.append(density_map)
    
    if not images_list:
        print(f"Warning: No valid data for split {split}.")
        continue
    
    # Stack all images and labels into numpy arrays
    images_array = np.stack(images_list, axis=0)
    labels_array = np.stack(labels_list, axis=0)
    
    # Save to HDF5 with appropriate data types (float32)
    h5_path = os.path.join(output_dir, f'{split}.h5')
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('images', data=images_array, dtype='float32', 
                          compression="gzip", compression_opts=4)
        hf.create_dataset('labels', data=labels_array, dtype='float32', 
                          compression="gzip", compression_opts=4)
    
    print(f"Created {h5_path} with {len(images_list)} samples.")

print("Dataset adaptation complete.")
