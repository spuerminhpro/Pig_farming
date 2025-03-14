import os
import glob
import cv2
import numpy as np
import random
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def generate_density_map(image_shape, dot_annotations, sigma=15):
    "Tạo density map"
    height, width = image_shape
    density_map = np.zeros((height, width), dtype=np.float32) #Tạo mảng 0 theo kích thước của ảnh
    for x, y in dot_annotations: #Duyệt qua kiểm các điểm center của bounding box
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height: 
            density_map[y, x] = 1.0
    density_map = gaussian_filter(density_map, sigma=sigma) #Sử dụng Gaussian filter để tạo map
    return density_map

def select_exemplars(boxes, num_exemplars=3):
    " Chọn ra 3 bounding box để làm exemplar"
    if len(boxes) <= num_exemplars:
        return boxes
    return random.sample(boxes, num_exemplars)

def extract_boxes_from_track(track, frame_id):
    boxes = []
    for tracklet in track:
        for data in tracklet:
            if data[0] == frame_id:
                bbox = [float(val) for val in data[1:5]]  # bbox format: [x1, y1, x2, y2]
                boxes.append(bbox)
    return boxes

def boxes_to_centers(boxes):
    "Lấy điểm center box"
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append([center_x, center_y])
    return centers

def add_random_padding_to_frame(frame, scale_factor=0.45): #giảm còn 45%
    if scale_factor is None:     # scale_factor = 100%   
        return frame, 0, 0, 1.0  # No padding
    
    orig_h, orig_w = frame.shape[:2]
    new_w = int(orig_w * scale_factor) 
    new_h = int(orig_h * scale_factor) 
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tổng size padding cần khôi phục size ảnh gốc
    total_pad_w = orig_w - new_w
    total_pad_h = orig_h - new_h
    
    # Random tỉ lệ padding
    fractions = random.randrange(1, 100)
    pad_top = int(fractions / 100 * total_pad_h)
    pad_left = int(fractions / 100 * total_pad_w)
    pad_bottom = total_pad_h - pad_top
    pad_right = total_pad_w - pad_left
    
    # Add padding
    padded_frame = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom,
                                      pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_frame, pad_left, pad_top, scale_factor

def adjust_boxes_and_points(boxes, points, pad_left, pad_top, scale_factor):
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box  # box từ ảnh gốc
        # Scale coordinates
        x1_scaled = x1 * scale_factor
        y1_scaled = y1 * scale_factor
        x2_scaled = x2 * scale_factor
        y2_scaled = y2 * scale_factor
        # Shift by padding
        x1_padded = x1_scaled + pad_left
        y1_padded = y1_scaled + pad_top
        x2_padded = x2_scaled + pad_left
        y2_padded = y2_scaled + pad_top
        adjusted_boxes.append([x1_padded, y1_padded, x2_padded, y2_padded])
    
    adjusted_points = []
    for point in points:
        px, py = point # center point từ ảnh gốc
        # Scale coordinates
        px_scaled = px * scale_factor
        py_scaled = py * scale_factor
        # Shift by padding
        px_padded = px_scaled + pad_left
        py_padded = py_scaled + pad_top
        adjusted_points.append([px_padded, py_padded])
    
    return adjusted_boxes, adjusted_points

# Utility functions (assumed available)
from create_dataset_utils import (
    generate_density_map,
    select_exemplars,
    extract_boxes_from_track,
    boxes_to_centers,
    add_random_padding_to_frame,
    adjust_boxes_and_points
)

def create_dataset_for_video(video_file, track_file, images_output_dir, density_maps_output_dir, annotations, video_id, global_frame_counter,scales=[None, 0.45, 0.55]):
    """
    Creates a dataset from a video, generating three versions per frame: original, scale 0.45, and scale 0.55.

    Args:
        video_file (str): Path to the video file.
        track_file (str): Path to the track file with annotations.
        images_output_dir (str): Directory to save images.
        density_maps_output_dir (str): Directory to save density maps.
        annotations (dict): Dictionary to store annotations.
        video_id (str): Identifier for the video.
        global_frame_counter (int): Counter for unique frame numbering.

    Returns:
        tuple: Updated global_frame_counter and annotations dictionary.
    """
    track_data = np.load(track_file, allow_pickle=True)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video file {video_file}")
        return global_frame_counter, annotations
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"No frames found in video file {video_file}")
        cap.release()
        return global_frame_counter, annotations

    # Extract 10 evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)

    for frame_id in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_id} from video {video_file}")
            continue

        boxes = extract_boxes_from_track(track_data, frame_id)
        dot_annotations = boxes_to_centers(boxes)
        frame_number = global_frame_counter  #

        # Process scale
        for scale in scales:
            if scale is None:
                scale_str = "original"
                padded_frame = frame
                adjusted_boxes = boxes
                adjusted_points = dot_annotations
            else:
                scale_str = f"scale{int(scale * 100):03d}" 
                padded_frame, pad_left, pad_top, scale_factor = add_random_padding_to_frame(frame, scale)
                adjusted_boxes, adjusted_points = adjust_boxes_and_points(boxes, dot_annotations, pad_left, pad_top, scale_factor)

            # filenames
            image_filename = f"{frame_number:06d}_{scale_str}.jpg"
            image_path = os.path.join(images_output_dir, image_filename)
            cv2.imwrite(image_path, padded_frame)

            # Tạo density map và lưu
            image_shape = padded_frame.shape[:2]
            density_map = generate_density_map(image_shape, adjusted_points)
            density_filename = f"{frame_number:06d}_{scale_str}.npy"
            density_path = os.path.join(density_maps_output_dir, density_filename)
            np.save(density_path, density_map)

            # Update annotations
            exemplar_boxes = select_exemplars(adjusted_boxes, num_exemplars=3)
            exemplar_coords = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in exemplar_boxes]
            annotations[image_filename] = {
                "H": image_shape[0],
                "W": image_shape[1],
                "box_examples_coordinates": exemplar_coords,
                "points": adjusted_points
            }
            print(f"Processed image {image_filename} from video {video_id} (frame {frame_id})")

        global_frame_counter += 1

    cap.release()
    return global_frame_counter, annotations

def create_dataset_for_image(image, boxes, images_output_dir, density_maps_output_dir, annotations, global_frame_counter, scales=[None, 0.45, 0.55]):
    # Convert boxes to center points using the utility function from create_dataset_utils
    dot_annotations = boxes_to_centers(boxes)
    frame_number = global_frame_counter  # Base name for this frame’s outputs

    for scale in scales:
        if scale is None:
            scale_str = "original"
            processed_image = image.copy()
            adjusted_boxes = boxes
            adjusted_points = dot_annotations
        else:
            scale_str = f"scale{int(scale * 100):03d}"  # e.g., "scale045" for a scale of 0.45
            processed_image, pad_left, pad_top, applied_scale = add_random_padding_to_frame(image, scale)
            adjusted_boxes, adjusted_points = adjust_boxes_and_points(boxes, dot_annotations, pad_left, pad_top, applied_scale)

        # Define filenames
        image_filename = f"{frame_number:06d}_{scale_str}.jpg"
        image_path = os.path.join(images_output_dir, image_filename)
        cv2.imwrite(image_path, processed_image)

        # Generate and save the density map using the utility function
        image_shape = processed_image.shape[:2]
        density_map = generate_density_map(image_shape, adjusted_points)
        density_filename = f"{frame_number:06d}_{scale_str}.npy"
        density_path = os.path.join(density_maps_output_dir, density_filename)
        np.save(density_path, density_map)

        # Update annotations with exemplar boxes for the current image
        exemplar_boxes = select_exemplars(adjusted_boxes, num_exemplars=3)
        exemplar_coords = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in exemplar_boxes]
        annotations[image_filename] = {
            "H": image_shape[0],
            "W": image_shape[1],
            "box_examples_coordinates": exemplar_coords,
            "points": adjusted_points
        }
        print(f"Processed image {image_filename}")

    # Increment counter after processing all scales for this image
    global_frame_counter += 1

    return global_frame_counter, annotations



def create_data_split(image_dir, output_dir, train_ratio=0.6, val_ratio=0.2):
    # Lấy tên ảnh từ thư mục
    image_names = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: x.split('.')[0]
    )
    
    # Tính số lượng ảnh cho train_set, val_set, test_set
    total_images = len(image_names)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Bởi vì ảnh extract 10 frame/video nên cần chia theo thứ tự để tránh trùng video frame trên train_set, val_set, test_set
    train_images = image_names[:train_end]  # 70% sô ảnh đầu tiên cho training
    val_images = image_names[train_end:val_end]  # 15% ảnh tiếp theo cho validation
    test_images = image_names[val_end:]  # 15% ảnh còn lại cho testing
    
    # Tạo dict cho data split
    data_split = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }
    
    # Save in JSON file
    split_file = os.path.join(output_dir, 'data_split.json')
    with open(split_file, 'w') as f:
        json.dump(data_split, f, indent=4)
    
    print(f"Data split saved to {split_file}")
