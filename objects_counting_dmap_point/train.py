"""Main script used to train networks."""
import os
from typing import Union, Optional, List

import click
import torch
import numpy as np
from matplotlib import pyplot
import re
import glob
from data_loader import H5Dataset
from looper import Looper,CombinedCountingLoss
from model import UNet, FCRN_A
from resnet50 import CountingModel

@click.command()
@click.option('-d', '--dataset_name',
              type=click.Choice(['cell', 'mall', 'ucsd', 'pig_farming']),
              required=True,
              help='Dataset to train model on (expect proper HDF5 files).')
@click.option('-n', '--network_architecture',
              type=click.Choice(['UNet', 'FCRN_A', 'CountingModel']),
              required=True,
              help='Model to train.')
@click.option('-lr', '--learning_rate', default=2e-5,
              help='Initial learning rate (lr_scheduler is applied).')
@click.option('-e', '--epochs', default=150, help='Number of training epochs.')
@click.option('--batch_size', default=8,
              help='Batch size for both training and validation dataloaders.')
@click.option('-hf', '--horizontal_flip', default=0.0,
              help='The probability of horizontal flip for training dataset.')
@click.option('-vf', '--vertical_flip', default=0.0,
              help='The probability of horizontal flip for validation dataset.')
@click.option('--unet_filters', default=64,
              help='Number of filters for U-Net convolutional layers.')
@click.option('--convolutions', default=2,
              help='Number of layers in a convolutional block.')
@click.option('--plot', is_flag=True, help="Generate a live plot.")
def train(dataset_name: str,
          network_architecture: str,
          learning_rate: float,
          epochs: int,
          batch_size: int,
          horizontal_flip: float,
          vertical_flip: float,
          unet_filters: int,
          convolutions: int,
          plot: bool):
    """Train chosen model on selected dataset."""
    # use GPU if avilable
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = {}     # training and validation HDF5-based datasets
    dataloader = {}  # training and validation dataloaders

    for mode in ['train', 'valid']:
        # expected HDF5 files in dataset_name/(train | valid).h5
        data_path = os.path.join('/mnt/sda1/PythonProject/Pig_counting/Pig_farming/objects_counting_dmap_point/data/pig_detection', f"{mode}.h5")

        # turn on flips only for training dataset
        dataset[mode] = H5Dataset(data_path,
                                  horizontal_flip if mode == 'train' else 0,
                                  vertical_flip if mode == 'train' else 0)
        dataloader[mode] = torch.utils.data.DataLoader(dataset[mode],
                                                       batch_size=batch_size)

    # only UCSD dataset provides greyscale images instead of RGB
    input_channels = 1 if dataset_name == 'ucsd' else 3
    
    # initialize a model based on chosen network_architecture
    if network_architecture == 'CountingModel':
        network = CountingModel(pool='mean').to(device)
    else:
        network = {
            'UNet': UNet,
            'FCRN_A': FCRN_A,
        }[network_architecture](input_filters=input_channels,
                            filters=unet_filters,
                            N=convolutions).to(device)
    network = torch.nn.DataParallel(network)

    # Load checkpoint if exists
    checkpoint_dir = 'checkpoints'
    checkpoint_pattern = f'{dataset_name}_{network_architecture}_epoch*.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_pattern)
    
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
        checkpoint_files = []
    else:
        # Find all matching checkpoint files
        checkpoint_files = glob.glob(checkpoint_path)
    
    if checkpoint_files:
        # Extract epoch numbers and find the latest
        latest_checkpoint = max(checkpoint_files, 
                              key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))
        
        # Load the latest checkpoint
        network.load_state_dict(torch.load(latest_checkpoint))
        print(f"Latest checkpoint file {latest_checkpoint} loaded successfully.")
    else:
        print(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
    # initialize loss, optimized and learning rate scheduler
    loss =  CombinedCountingLoss()# Increase beta to give more weight to count loss
    optimizer = torch.optim.Adam(network.parameters(),
                                lr=2e-5,  # Lower learning rate for better stability
                                weight_decay=1e-5)  # Reduce weight decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)


    # if plot flag is on, create a live plot (to be updated by Looper)
    if plot:
        pyplot.ion()
        fig, plots = pyplot.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2

    # create training and validation Loopers to handle a single epoch
    train_looper = Looper(network, device, loss, optimizer,
                          dataloader['train'], len(dataset['train']), plots[0])
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader['valid'], len(dataset['valid']), plots[1],
                          validation=True)

    # current best results (lowest mean absolute error on validation set)
    current_best = np.inf
    #create a folder to save the eval metrics
    os.makedirs('eval_metrics', exist_ok=True)
    # Load the last epoch from eval_metrics.json
    last_epoch = 0
    if os.path.exists('eval_metrics/eval_metrics.json'):
        with open('eval_metrics/eval_metrics.json', 'r') as f:
            lines = f.readlines()
            if lines:
                # Get the last line and extract the epoch number
                last_line = lines[-1]
                last_epoch = int(last_line.split()[0])   # Start from next epoch
    
    for epoch in range(last_epoch, epochs):
        print(f"Epoch {epoch}\n")
        # run training epoch and update learning rate
        train_looper.run(epoch)
        lr_scheduler.step()

        # run validation epoch
        with torch.no_grad():
            result = valid_looper.run(epoch)
        print(f"Validation result: {result}")
        # update checkpoint if new best is reached
        if result < current_best:
            current_best = result
            # Create checkpoint directory if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)
            # Save best model with epoch number
            torch.save(network.state_dict(), 
                      os.path.join('checkpoints', f'{dataset_name}_{network_architecture}_epoch{epoch}.pth'))
            #save final
            print(f"\nNew best result: {result}")

        print("\n", "-"*80, "\n", sep='')
    # Save final checkpoint
    torch.save(network.state_dict(), 
              os.path.join('checkpoints', f'{dataset_name}_{network_architecture}_final_epoch{epochs}.pth'))
    
    print(f"[Training done] Best result: {current_best}")

if __name__ == '__main__':
    train()
