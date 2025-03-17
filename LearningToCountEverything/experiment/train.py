import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DirectCountingModel
from data import SingleClassDataset
from utils import TransformTrain
import os
import json
from tqdm import tqdm
import torch.nn.functional as F

data_path = '/content/'
train_dataset = SingleClassDataset(data_path=data_path, split='train', target_class='car', transform=TransformTrain)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

model = DirectCountingModel().cuda()
criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 5
output_dir = './training_results'
os.makedirs(output_dir, exist_ok=True)
results_file = os.path.join(output_dir, 'training_results.json')

training_results = []

def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    train_mae = 0.0
    train_rmse = 0.0
    cnt = 0
    epoch_results = {'epoch': epoch + 1, 'images': []}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        images = batch['image'].cuda()
        gt_density = batch['gt_density'].cuda()

        optimizer.zero_grad()
        pred_density = model(images)

        if len(pred_density.shape) != 4:
            pred_density = pred_density.view(1, 1, *pred_density.shape[-2:])
        if len(gt_density.shape) != 4:
            gt_density = gt_density.view(1, 1, *gt_density.shape[-2:])

        # Interpolate to match gt_density's spatial dimensions
        target_size = (gt_density.shape[2], gt_density.shape[3])
        pred_density = F.interpolate(pred_density, size=target_size, mode='bilinear', align_corners=False)

        # MSE loss for density map
        mse_loss = criterion(pred_density, gt_density)
        # Count loss for regularization
        gt_cnt = torch.sum(gt_density)
        pred_cnt = torch.sum(pred_density)
        count_loss = F.mse_loss(pred_cnt, gt_cnt)
        total_loss = mse_loss + 0.1 * count_loss  # Adjust weight as needed

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        cnt_err = abs(pred_cnt.item() - gt_cnt.item())
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        cnt += 1

        pbar.set_description(
            f"actual-predicted: {gt_cnt.item():6.1f}, {pred_cnt.item():6.1f}, error: {cnt_err:6.1f}. "
            f"Current MAE: {train_mae/cnt:5.2f}, RMSE: {(train_rmse/cnt)**0.5:5.2f}"
        )

        epoch_results['images'].append({
            'image_id': train_dataset.im_ids[cnt-1],
            'actual_count': gt_cnt.item(),
            'predicted_count': pred_cnt.item(),
            'error': cnt_err
        })

    avg_loss = running_loss / len(train_loader)
    avg_mae = train_mae / len(train_loader)
    avg_rmse = (train_rmse / len(train_loader)) ** 0.5

    epoch_results.update({'loss': avg_loss, 'mae': avg_mae, 'rmse': avg_rmse})
    return epoch_results

for epoch in range(num_epochs):
    epoch_results = train_epoch(epoch)
    training_results.append(epoch_results)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_results['loss']:.4f}, "
          f"MAE: {epoch_results['mae']:.2f}, RMSE: {epoch_results['rmse']:.2f}")
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=4)

torch.save(model.state_dict(), os.path.join(output_dir, 'trained_direct_counting_model.pth'))
print(f"Training completed. Results saved to {results_file}")