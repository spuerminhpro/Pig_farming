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



def create_dataset(video_file, track_file, images_output_dir, density_maps_output_dir, annotations, video_id, global_frame_counter, scale_factor=None):    
    track_data = np.load(track_file, allow_pickle=True)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video file {video_file}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"No frames found in video file {video_file}")
        cap.release()
    else:
        # chia 10 frame từ video
        frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int) # chia thành 10 giá trị có khoảng cách bằng nhau
        for frame_id in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_id} from video {video_file}")
                continue
            image_filename = f"{global_frame_counter}.jpg"
            image_path = os.path.join(images_output_dir, image_filename)
                        
            boxes = extract_boxes_from_track(track_data, frame_id)
            dot_annotations = boxes_to_centers(boxes)

            if scale_factor is not None: # scale size ảnh
                padded_frame, pad_left, pad_top, scale = add_random_padding_to_frame(frame, scale_factor)
                adjusted_boxes, adjusted_points = adjust_boxes_and_points(boxes, dot_annotations, pad_left, pad_top, scale)
                cv2.imwrite(image_path, padded_frame)
            else:                        # giữ nguyên size ảnh
                adjusted_boxes = boxes
                adjusted_points = dot_annotations
                cv2.imwrite(image_path, frame)
                
        
            # Tạo density map
            image_shape = (padded_frame.shape[0], padded_frame.shape[1])
            density_map = generate_density_map(image_shape, adjusted_points)
            density_filename = f"{global_frame_counter}.npy"
            density_path = os.path.join(density_maps_output_dir, density_filename)
            np.save(density_path, density_map)
            
            # Lấy 3 bounding box để làm exemplar
            exemplar_boxes = select_exemplars(adjusted_boxes, num_exemplars=3)
            
            # Lưu các điểm của bounding box
            exemplar_coords = []
            for box in exemplar_boxes:
                x1, y1, x2, y2 = box
                # Model format input: [x1, y1], [x1, y2], [x2, y2], [x2, y1]
                exemplar_coords.append([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]) 
            
            # Lưu annotations
            annotations[image_filename] = {
                "H": image_shape[0],
                "W": image_shape[1],
                "box_examples_coordinates": exemplar_coords,
                "points": adjusted_points
            }
            
            print(f"Processed frame {global_frame_counter} from video {video_id} (video frame id: {frame_id})")
            global_frame_counter += 1

        cap.release()

    return global_frame_counter, annotations

import os
import json

def create_data_split(image_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    # Lấy tên ảnh từ thư mục
    image_names = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])
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
