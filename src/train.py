import os
import gc
import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import random_split
import yaml

import models
from dataset import BaseDataset # BaseDataset is the correct class now
from tools import train
from utils import filter_data_from_h5



model_dict = {
    'CNN_GRU': models.CNN_GRU,
    'CNN_BiGRU': models.CNN_BiGRU,
    'CNN': models.CNN,
    'CNN_LSTM': models.CNN_LSTM,
    'TCN': models.TCN,
    'MESTNet': models.MESTNet
}
        
2       
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Trains a Neural Network archtiecture under a non-ideal condition factor."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True, 
        help="The directory to save out the results of the training process."
    )
    parser.add_argument(
        '--config_file', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file.'
    )

    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    config_file_path = os.path.abspath(args.config_file)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created output directory: {result_dir}")
    elif not os.path.isdir(result_dir):
        print(f"Error: Output path exists but is not a directory: {result_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at {config_file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_file_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded configuration from {config_file_path}")
        # Basic validation of config structure
        required_sections = ['dataset', 'training', 'optimizer']
        for section in required_sections:
            if section not in cfg:
                raise ValueError(f"Missing required section '{section}' in config file.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}", file=sys.stderr)
        sys.exit(1)

    
    data_file = cfg['dataset']['data_file']  # Path to the dataset file

    subs = cfg['filter']['specific_subjects']
    sequence_length = cfg['dataset']['sequence_length']
    num_classes = cfg['dataset']['classes']
    channels = cfg['dataset']['channels']

    experiment_dir = os.path.join(result_dir, cfg['experiment']['factor'])

    model_dir = os.path.join(experiment_dir, cfg['model']['name'])

    os.makedirs(model_dir, exist_ok=True)
  
    for r in range(3):

        loss = nn.CrossEntropyLoss()

        model_name = cfg['model']['name']

        if model_name == 'MESTNet':
            model = models.MESTNet(
                in_channels=cfg['dataset']['channels'], 
                num_classes=cfg['dataset']['classes'], 
                scales=cfg['dataset']['scales'],
                wavelet=cfg['dataset']['wavelet'],
                loss=loss
            )
        else:
            model = model_dict[model_name](
                in_channels=cfg['dataset']['channels'], 
                num_classes=cfg['dataset']['classes'],
                loss=loss
            )
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            cfg['training']['lr'], 
            weight_decay=cfg['finetuning']['weight_decay']
        )
    
        model.configure_optimizers = lambda: optimizer
        
        #compiled_model = torch.compile(current_model)
        print(f"Rep. with test set r={r}")

        # repetitions to include for training
        reps = [j for j in range(3) if j != r]

        train_idxs = filter_data_from_h5(
            data_file=data_file, 
            positions = cfg['filter']['positions'],
            sessions = cfg['filter']['sessions'],
            repetitions=reps,
            subjects = cfg['filter']['specific_subjects']
        )

        full_train_dataset = BaseDataset(
            data_file,
            idxs=train_idxs,
            load_in_memory=True
        )

        # 80 - 20 train-val split
        train_size = int(cfg['training']['val_split'] / 100 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        train_data, val_data = random_split(full_train_dataset, [train_size, val_size])

        # make dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=cfg['training']['batch_size'], 
            shuffle=True, 
            num_workers=cfg['training']['num_workers']
        )

        val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=cfg['validation']['batch_size'], 
            shuffle=False, 
            num_workers=cfg['validation']['num_workers']
        )

        test_idxs = filter_data_from_h5(
            data_file=data_file, 
            positions = cfg['filter']['positions'],
            sessions = cfg['filter']['sessions'],
            repetitions=r,
            subjects = cfg['filter']['specific_subjects']
        )
        
        test_dataset = BaseDataset(
            data_file,
            idxs=test_idxs,
            load_in_memory=True,
            subset = 0.2
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=cfg['training']['val_batch_size'], 
            shuffle=False, 
            num_workers=cfg['training']['num_workers']
        )

        trained_model_path = train(
            model, 
            train_loader=train_loader, 
            val_loaders=[val_loader, test_loader], 
            reference_dir=model_dir,
            repetition=r,
            cfg = cfg
        )