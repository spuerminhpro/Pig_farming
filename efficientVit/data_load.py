import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CocoSegDataset(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        labels = []
        masks = []
        for ann in anns:
            bboxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        bboxes = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.empty((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        masks = np.array(masks) if masks else np.zeros((0, img_info['height'], img_info['width']), dtype=np.uint8)
        masks = torch.tensor(masks, dtype=torch.uint8)
        if self.transform:
            img = self.transform(img)
            if masks.size(0) > 0:
                masks = T.Resize((80, 80), interpolation=T.InterpolationMode.NEAREST)(masks)
        return {
            'img': img,
            'gt_bboxes': bboxes,
            'gt_labels': labels,
            'gt_masks': masks,
            'image_id': img_id  # Added image_id
        }

def get_transforms(is_train=True):
    """Define image transformations."""
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(640, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            normalize
        ])
    return normalize

def build_data_loaders(train_dataset,val_dataset,batch_size=2, num_workers=2):
    """Build train and validation DataLoaders."""

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    return {
        'img': torch.stack([item['img'] for item in batch]),
        'gt_bboxes': [item['gt_bboxes'] for item in batch],
        'gt_labels': [item['gt_labels'] for item in batch],
        'gt_masks': [item['gt_masks'] for item in batch],
        'image_id': [item['image_id'] for item in batch]
    }

def instance_loss_fn(embeddings, gt_masks, batch_size):
    loss = 0
    for b in range(batch_size):
        emb = embeddings[b].permute(1, 2, 0)
        masks = gt_masks[b]
        if len(masks) <= 1: continue
        emb_flat = emb.reshape(-1, emb.size(-1))
        mask_flat = masks.reshape(len(masks), -1)
        instance_emb = []
        for i in range(len(masks)):
            mask_pixels = emb_flat[mask_flat[i] > 0]
            if mask_pixels.numel() > 0:
                instance_emb.append(mask_pixels.mean(dim=0))
            else:
                instance_emb.append(torch.zeros(emb_flat.size(-1), device=device))
        if len(instance_emb) <= 1: continue
        instance_emb = torch.stack(instance_emb)
        for i in range(len(instance_emb)):
            for j in range(i + 1, len(instance_emb)):
                diff = instance_emb[i] - instance_emb[j]
                loss += torch.relu(1.0 - diff.norm()).pow(2)
    return loss / batch_size if loss > 0 else loss

def save_and_plot_metrics(epoch, avg_loss, mAP, metrics_file="checkpoints/training_metrics.json", save_dir="checkpoints"):
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {'train_losses': [], 'val_mAPs': []}
    metrics['train_losses'].append(float(avg_loss))
    metrics['val_mAPs'].append(float(mAP if mAP is not None else 0.0))
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    num_epochs = len(metrics['train_losses'])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), metrics['train_losses'], label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), metrics['val_mAPs'], label='Validation mAP', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"training_metrics_epoch_{epoch+1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def load_checkpoint(model, optimizer, checkpoint_path='checkpoints/efficientvit_pig_seg_final.pth', device=device):
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    raise KeyError("Checkpoint does not contain 'model_state_dict'")
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    print("Warning: 'optimizer_state_dict' not found in checkpoint, optimizer will use fresh state")
                start_epoch = checkpoint.get('epoch', -1) + 1
                print(f"Resuming from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}, starting at epoch {start_epoch}")
                return start_epoch
            else:
                # Handle raw state dict (older checkpoints)
                model.load_state_dict(checkpoint)
                print(f"Loaded raw model state from {checkpoint_path}, starting from epoch 0 (no optimizer state)")
                return 0
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            return 0
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting training from scratch")
        return 0

