import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import random

'''set your data path'''
root = '/mnt/sda1/PythonProject/Pig_counting/dataset/PigsFarming_label_dataset'
output_dir = '/mnt/sda1/PythonProject/Pig_counting/dataset/TransCrowd/'
train_images = os.path.join(root, 'train', 'images')
test_images = os.path.join(root, 'test', 'images')

path_sets = [train_images, test_images]

'''for train'''
if not os.path.exists(os.path.join(root, 'train/gt_density_map_crop')):
    os.makedirs(os.path.join(root, 'train/gt_density_map_crop'))

if not os.path.exists(os.path.join(root, 'train/images_crop')):
    os.makedirs(os.path.join(root, 'train/images_crop'))

'''for test'''
if not os.path.exists(os.path.join(root, 'val_test/gt_density_map_crop')):
    os.makedirs(os.path.join(root, 'val_test/gt_density_map_crop'))

if not os.path.exists(os.path.join(root, 'val_test/images_crop')):
    os.makedirs(os.path.join(root, 'val_test/images_crop'))

if not os.path.exists(os.path.join(output_dir, 'train/images_crop')):
    os.makedirs(os.path.join(output_dir, 'train/images_crop'))

if not os.path.exists(os.path.join(output_dir, 'val_test/images_crop')):
    os.makedirs(os.path.join(output_dir, 'val_test/images_crop'))

if not os.path.exists(os.path.join(output_dir, 'train/gt_density_map_crop')):
    os.makedirs(os.path.join(output_dir, 'train/gt_density_map_crop'))

if not os.path.exists(os.path.join(output_dir, 'val_test/gt_density_map_crop')):
    os.makedirs(os.path.join(output_dir, 'val_test/gt_density_map_crop'))



def read_yolo_boxes(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            # Convert from normalized to absolute coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
    return boxes

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
img_paths.sort()

np.random.seed(0)
random.seed(0)

for img_path in img_paths:
    print('processing ', img_path)
    img = cv2.imread(img_path)
    original_height, original_width = img.shape[:2]

    is_train = 'train' in img_path

    label_file = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    if not os.path.exists(label_file):
        print(f'Warning: Label file {label_file} does not exist, skipping image')
        continue

    boxes = read_yolo_boxes(label_file, original_width, original_height)
    points = [(int((box[0] + box[2])/2), int((box[1] + box[3])/2)) for box in boxes]

    # Resize image and adjust point coordinates
    rate = 1
    rate_1 = 1
    rate_2 = 1
    
    if original_width >= original_height:  # width is larger
        rate_1 = 1152.0 / original_width
        rate_2 = 768.0 / original_height
        img_resized = cv2.resize(img, (0, 0), fx=rate_1, fy=rate_2)
        # Adjust point coordinates
        points = [(int(p[0] * rate_1), int(p[1] * rate_2)) for p in points]
    else:  # height is larger
        rate_1 = 1152.0 / original_height
        rate_2 = 768.0 / original_width
        img_resized = cv2.resize(img, (0, 0), fx=rate_2, fy=rate_1)
        # Adjust point coordinates
        points = [(int(p[0] * rate_2), int(p[1] * rate_1)) for p in points]
        print(img_path)

    # Create density map
    height, width = img_resized.shape[0], img_resized.shape[1]
    kpoint = np.zeros((height, width))
    
    for point in points:
        if point[0] < width and point[1] < height:
            kpoint[point[1], point[0]] = 1

    if is_train:
        # Calculate number of crops based on image dimensions
        m = int(width / 384)
        n = int(height / 384)
        fname = os.path.basename(img_path)
        root_path = os.path.join(output_dir, 'train/images_crop')
        
        for i in range(0, m):
            for j in range(0, n):
                crop_img = img_resized[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384]
                gt_count = np.sum(crop_kpoint)

                save_fname = str(i) + str(j) + str('_') + fname
                save_path = os.path.join(root_path, save_fname)

                h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
                if gt_count == 0:
                    print(save_path, h5_path)
                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count
                    # Store the points for visualization
                    hf['points'] = np.array(points)

                cv2.imwrite(save_path, crop_img)
    else:
        # For test images, save the full image
        save_path = os.path.join(output_dir, 'val_test/images_crop', os.path.basename(img_path))
        cv2.imwrite(save_path, img_resized)

        gt_count = np.sum(kpoint)
        h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
        with h5py.File(h5_path, 'w') as hf:
            hf['gt_count'] = gt_count
            # Store the points for visualization
            hf['points'] = np.array(points)

print("Dataset preparation completed.")