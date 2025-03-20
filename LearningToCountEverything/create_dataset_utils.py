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
    if points is not None:
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

def create_dataset_for_video_FSC147(video_file, track_file, images_output_dir, density_maps_output_dir, annotations, video_id, global_frame_counter,scales=[None, 0.45, 0.55]):
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

def create_dataset_for_image_FSC147(image, boxes, images_output_dir, density_maps_output_dir, annotations, global_frame_counter, scales=[None, 0.45, 0.55]):
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

def create_dataset_GroundDino_odvg(image, boxes,categories, output_dir, annotations, global_frame_counter, scales=[None, 0.45, 0.55]):
    frame_number = global_frame_counter

    for scale in scales:
        if scale is None:
            scale_str = "original"
            processed_image = image.copy()
            adjusted_boxes = boxes
        else:
            scale_str = f"scale{int(scale * 100):03d}"
            processed_image, pad_left, pad_top, applied_scale = add_random_padding_to_frame(image, scale)
            adjusted_boxes, _ = adjust_boxes_and_points(boxes, [], pad_left, pad_top, applied_scale)

        image_filename = f"{frame_number:06d}_{scale_str}.jpg"
        images_output_dir = os.path.join(output_dir, "images")
        image_path = os.path.join(images_output_dir, image_filename)
        cv2.imwrite(image_path, processed_image)
        image_shape = processed_image.shape[:2]

        exemplar_boxes = select_exemplars(adjusted_boxes, num_exemplars=3)
        #Convert exemplar boxes to the format in the example jsonl
        exemplar_coords = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in exemplar_boxes]


        annotations.append({  # Append to a list, not a dictionary
            "filename": image_filename,
            "height": image_shape[0],
            "width": image_shape[1],
            "detection": {
                "instances": [
                    {
                        "bbox": [(x1), (y1), (x2), (y2)],  
                        "label": 0,  # id category
                        "category": categories["categories"][0]['name'] #class
                    } for x1, y1, x2, y2 in adjusted_boxes
                ]
            },
            "exemplars": exemplar_coords
        })
        print(f"Processed image {image_filename}")

    global_frame_counter += 1

    return global_frame_counter, annotations



def create_dataset_GroundDino_coco(image_paths, boxes_list, output_dir, categories, global_frame_counter, scales=[None, 0.45, 0.55]):
    """
    Builds a COCO-formatted dataset using the provided images and bounding boxes, including scaled versions.

    Args:
        image_paths (list): List of paths to the original images.
        boxes_list (list): List of lists, where each inner list contains bounding boxes [x1, y1, x2, y2] for the corresponding image.
        output_dir (str): Base directory to save the processed images.
        categories (dict): A dictionary with a "categories" key containing a list of category dictionaries.
        global_frame_counter (int): Starting counter for unique filenames.
        scales (list): List of scales to apply, where None means original.

    Returns:
        tuple: A tuple (coco_dataset, new_global_frame_counter) where coco_dataset is a dictionary with 'images', 'annotations',
               and 'categories' in COCO format and new_global_frame_counter is the updated frame counter.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define final images output directory
    images_output_dir_final = os.path.join(output_dir, "images")
    if not os.path.exists(images_output_dir_final):
        os.makedirs(images_output_dir_final)
    
    # Initialize COCO dataset components
    images_list = []
    annotation_list = []
    categories_list = categories  # This remains as the same dictionary format passed in.
    image_id = 1
    annotation_id_counter = 1
    frame_counter = global_frame_counter  # Use global_frame_counter as starting point

    # Process each image and its bounding boxes
    for image_path, boxes in zip(image_paths, boxes_list):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue

        # Apply each scale to the current image
        for scale in scales:
            if scale is None:
                scale_str = "original"
                processed_image = image.copy()
                adjusted_boxes = boxes
            else:
                scale_str = f"scale{int(scale * 100):03d}"
                processed_image, pad_left, pad_top, applied_scale = add_random_padding_to_frame(image, scale)
                adjusted_boxes, _ = adjust_boxes_and_points(boxes, None, pad_left, pad_top, applied_scale)

            # Define unique filename using frame_counter
            image_filename = f"{frame_counter:06d}_{scale_str}.jpg"
            output_path = os.path.join(images_output_dir_final, image_filename)
            cv2.imwrite(output_path, processed_image)
            image_shape = processed_image.shape[:2]  # (height, width)

            # Add image entry to images_list
            images_list.append({
                "height": image_shape[0],
                "width": image_shape[1],
                "id": image_id,
                "file_name": image_filename
            })

            # Add annotations for this image version
            for box in adjusted_boxes:
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                annotation_list.append({
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [float(x_min), float(y_min), int(width), int(height)],
                    "category_id":  categories["categories"][0]['id'],  # Assuming one category
                    "id": annotation_id_counter,
                    "area": float(area)
                })
                annotation_id_counter += 1

            print(f"Processed image {image_filename}")
            image_id += 1

        frame_counter += 1  # Increment for each base image

    # Update the global frame counter (could be returned for subsequent calls)
    global_frame_counter = frame_counter

    # Assemble the COCO dataset dictionary
    coco_dataset = {
        "images": images_list,
        "annotations": annotation_list,
        "categories": categories_list
    }

    return coco_dataset, global_frame_counter





    
        
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
