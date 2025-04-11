# COUNTGD: Multi-Modal Open-World Counting Dataset Preparation

## Dataset Structure
The dataset is structured as follows:

```
paperCountGD_Dataset/
├── Train_dataset/
│   ├── Image_file/
│   │   ├── Image_1.jpg
│   │   ├── Image_2.jpg
│   │   └── …
│   └── annotation.json
├── valid_dataset/
│   ├── Image_file/
│   │   ├── Image_1.jpg
│   │   ├── Image_2.jpg
│   │   └── …
│   └── annotation.json
└── label.json
```

## Dataset Construction from YOLOv8 Format
The YOLO label format needs to be converted into bounding box coordinates in the form `[x1, y1, x2, y2]`.

### Conversion Formula
YOLO label values are structured as follows:
```
<class_id> <x_center> <y_center> <width> <height>
```

Bounding box conversion:
```
x1 = x_center - width / 2  # min x coordinate
y1 = y_center - height / 2  # min y coordinate
x2 = x_center + width / 2  # max x coordinate
y2 = y_center + height / 2  # max y coordinate
```

**Output label format:** `[x1, y1, x2, y2]`

## ODVG Dataset Format
Since there is only one class (`pig`), the categories are structured accordingly.

### ODVG Annotation Format
Example `annotation.json`:
```json
{
    "filename": "000001_original.jpg",
    "height": 720,
    "width": 1280,
    "detection": {
        "instances": [
            {
                "bbox": [x1, y1, x2, y2],
                "label": 0,
                "category": "pig"
            },
            {
                "bbox": [x3, y3, x4, y4],
                "label": 0,
                "category": "pig"
            }
        ]
    },
    "exemplars": [
        [x1, y1, x2, y2],
        [x3, y3, x4, y4],
        [x5, y5, x6, y6]
    ]
}
```

Each image entry is followed by a newline character (`\n`) to separate records.

## COCO Dataset Format
The COCO annotation format includes `images`, `annotations`, and `categories` sections.

### COCO Annotation Format
Example `annotation.json`:

#### Images
```json
{
    "height": 720,
    "width": 1280,
    "id": 1,
    "file_name": "000001_original.jpg"
}
```

#### Annotations
```json
{
    "iscrowd": 0,
    "image_id": 1,
    "bbox": [x, y, width, height],  
    "category_id": 0,
    "id": 1,  
    "area": width * height
}
```
Each bounding box follows a global counter (`id += 1`).

#### Categories
```json
[
    {"id": 0, "name": "pig"}
]
```

## Label File
The `label.json` file defines class mappings:
```json
{
    "0": "pig"
}
```

This setup ensures compatibility with multiple annotation formats for object detection and counting tasks.

