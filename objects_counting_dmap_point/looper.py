"""Looper implementation."""
from typing import Optional, List

import torch
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch.nn as nn
import json
import os
# Improved Loss: combine pixel-wise MSE loss with global count L1 loss.
class CombinedCountingLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        """
        Args:
            alpha: weight for the density map reconstruction loss (MSE).
            beta: weight for the global count loss (L1).
        """
        super(CombinedCountingLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, gt):
        # Add small epsilon to prevent zero predictions
        epsilon = 1e-6
        pred = pred + epsilon
        gt = gt + epsilon
        
        # Pixel-wise density map loss
        loss_density = self.mse(pred, gt)
        
        # Global counting loss: difference between the sums of the predicted and ground truth maps.
        # Note: density maps are normalized by 100
        count_pred = torch.sum(pred, dim=[1,2,3]) 
        count_gt = torch.sum(gt, dim=[1,2,3]) 
        loss_count = self.l1(count_pred, count_gt)
        
        # Add debugging information
        if torch.isnan(loss_density) or torch.isnan(loss_count):
            print(f"Warning: NaN detected in loss calculation")
            print(f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"GT range: [{gt.min().item():.4f}, {gt.max().item():.4f}]")
            print(f"Count pred: {count_pred.mean().item():.4f}, Count gt: {count_gt.mean().item():.4f}")
        
        total_loss = self.alpha * loss_density + self.beta * loss_count
        return loss_density, loss_count, total_loss

loss_fn = nn.MSELoss()

def compute_loss(pred, target):
    target_sum = target.view(target.shape[0], -1).sum(dim=1, keepdim=True).view(target.shape[0], 1, 1, 1)  # Sum over spatial dims
    return loss_fn(pred, target_sum)

# Example during training:

class Looper():
    """Looper handles epoch loops, logging, and plotting."""

    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_size: int,
                 plots: Optional[matplotlib.axes.Axes]=None,
                 validation: bool=False):
        """
        Initialize Looper.

        Args:
            network: already initialized model
            device: a device model is working on
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            plot: matplotlib axes
            validation: flag to set train or eval mode

        """
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.plots = plots
        self.running_loss = []
        self.current_epoch = 0

    def _get_progress_bar(self, current, total, bar_length=50):
        """Create a progress bar string."""
        fraction = current / total
        filled_length = int(bar_length * fraction)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        return f'   [{bar}]    {current}/{total}'

    def run(self, epoch):
        """Run a single epoch loop.

        Returns:
            Mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.running_loss.append(0)
        current_loss = 0
        loss = 0
        rmse = 0
        self.current_epoch = epoch
        # set a proper mode: train or eval
        self.network.train(not self.validation)

        # Track total batches
        total_batches = len(self.loader)
        
        # Print initial progress bar
        print(f"\rEpoch {self.current_epoch} {self._get_progress_bar(0, total_batches)}", end='')
        
        for batch_idx, (image, label) in enumerate(self.loader):
            # Update progress bar
            # Initialize current_loss to 0 since loss hasn't been calculated yet
            
            

            
            # move images and labels to given device
            image = image.to(self.device)
            label = label.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            result = self.network(image)
        
            
            
            # calculate loss and update running loss
            loss_density, loss_count, total_loss = self.loss(result, label)
            
            # Add debugging information for zero predictions
            if torch.all(result == 0):
                print(f"\nWarning: Model predicted all zeros!")
                print(f"Input range: [{image.min().item():.4f}, {image.max().item():.4f}]")
                print(f"Label range: [{label.min().item():.4f}, {label.max().item():.4f}]")
            
            # calculate loss rmse
            rmse = torch.sqrt(total_loss)
            
            self.running_loss[-1] += image.shape[0] * rmse.item() / self.size

            # update weights if in train mode
            if not self.validation:
                total_loss.backward()
                self.optimizer.step()

            mean_true = 0
            mean_predicted = 0
            mean_error = 0
            # loop over batch samples
            for true, predicted in zip(label, result):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 100
                true_count = torch.sum(true).item()

                predicted_counts = torch.sum(predicted).item() / 100
                predicted_count = torch.sum(predicted).item()
                mean_true += true_count
                mean_predicted += predicted_count
                error= abs(true_count - predicted_count)
                mean_error += error
                print(f"True counts: {true_count:.2f}, Predicted counts: {predicted_count:.2f}")
                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)
            #caculate mean error
            mean_error = mean_error / len(label)
            print(f"\rEpoch {self.current_epoch} {self._get_progress_bar(batch_idx + 1, total_batches)}  Error: {mean_error:.2f} MSE Loss: {loss_density.item():.4f} MAE: {loss_count.item():.4f} RMSE: {rmse.item():.4f}", end='', flush=True)
        # calculate errors and standard deviation
        self.update_errors()

        # update live plot
        if self.plots is not None:
            self.plot()

        # print epoch summary
        self.log()
        #save eval metrics result
        result = {
            'mean_error': mean_error,
            'loss': total_loss,
            'MAE': loss_count,
            'RMSE': rmse
        }
        #check if eval_metrics.json exists
        if not os.path.exists('eval_metrics/eval_metrics.json'):
            with open('eval_metrics/eval_metrics.json', 'w') as f:
                f.write(f"{self.current_epoch} {result}\n")
        else:
            #add result to log by line
            with open('eval_metrics/eval_metrics.json', 'a') as f:
                f.write(f"{self.current_epoch} {result}\n")
        
        return self.mean_abs_err

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        self.err = [true - predicted for true, predicted in
                    zip(self.true_values, self.predicted_values)]
        self.abs_err = [abs(error) for error in self.err]
        self.mean_err = sum(self.err) / self.size
        self.mean_abs_err = sum(self.abs_err) / self.size
        self.std = np.array(self.err).std()

    def plot(self):
        """Plot true vs predicted counts and loss."""
        
            
        # true vs predicted counts
        true_line = [[0, max(self.true_values)]] * 2  # y = x
        self.plots[0].cla()
        self.plots[0].set_title('Train' if not self.validation else 'Valid')
        self.plots[0].set_xlabel('True value')
        self.plots[0].set_ylabel('Predicted value')
        self.plots[0].plot(*true_line, 'r-')
        self.plots[0].scatter(self.true_values, self.predicted_values)

        # loss
        epochs = np.arange(1, len(self.running_loss) + 1)
        self.plots[1].cla()
        self.plots[1].set_title('Train' if not self.validation else 'Valid')
        self.plots[1].set_xlabel('Epoch')
        self.plots[1].set_ylabel('Loss')
        self.plots[1].plot(epochs, self.running_loss)

        matplotlib.pyplot.pause(0.01)
        matplotlib.pyplot.tight_layout()

    def log(self):
        """Print current epoch results."""
        print(f"{'Train' if not self.validation else 'Valid'}:\n"
              f"\tAverage loss: {self.running_loss[-1]:3.4f}\n"
              f"\tMean error: {self.mean_err:3.3f}\n"
              f"\tMean absolute error: {self.mean_abs_err:3.3f}\n"
              f"\tError deviation: {self.std:3.3f}")
