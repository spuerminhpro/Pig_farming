import os
import json
import numpy as np
from PIL import Image
import torch
from utils import TransformTrain

class SingleClassDataset:
    def __init__(self, data_path, split, target_class=None, transform=None):
        self.data_path = data_path
        self.split = split  # 'train', 'val', or 'test'
        self.target_class = target_class  # Optional: can be None if not filtering by class
        self.transform = transform

        # Load data split
        with open(os.path.join(data_path, 'data_split.json')) as f:
            data_split = json.load(f)
        self.im_ids = data_split[split]

        # Comment out original class label loading since custom dataset has few classes
        # with open(os.path.join(data_path, 'ImageClasses_FSC147.txt')) as f:
        #     class_lines = f.readlines()
        #     self.image_classes = {line.split()[0]: line.split()[1] for line in class_lines}

        # For custom dataset
        if self.target_class:
            pass  
        else:
            pass

        # Load annotations
        with open(os.path.join(data_path, 'annotation.json')) as f:
            self.annotations = json.load(f)

        self.im_dir = os.path.join(data_path, 'images')
        self.gt_dir = os.path.join(data_path, 'gt_density_maps')

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_id = self.im_ids[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = []
        for bbox in bboxes:
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[2][0], bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open(os.path.join(self.im_dir, im_id))
        image.load()
        density_path = os.path.join(self.gt_dir, im_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')

        sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
        if self.transform:
            sample = self.transform(sample)

        return sample