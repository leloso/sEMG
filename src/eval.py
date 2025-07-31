import argparse
import os
import sys
import json

import yaml
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from dataset import BaseDataset
from utils import filter_data_from_h5, display_results

model_map = {
        'CNN': models.CNN,
        'CNN_GRU': models.CNN_GRU,
        'CNN_BiGRU': models.CNN_BiGRU,
        'CNN_LSTM': models.CNN_LSTM,
        'TCN': models.TCN,
        'MESTNet': models.MESTNet
    }



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Evaluates performance on multi factor condition."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True, 
        help="The directory containing trained models."
    )
    parser.add_argument(
        '--config_file', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '--result_dir', 
        type=str, 
        required=True, 
        help='Path to the directory where the evalutaion results will be stored'
    )

    args = parser.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    config_file_path = os.path.abspath(args.config_file)
    result_dir = os.path.abspath(args.result_dir)
    
    if not os.path.isdir(models_dir):
        print(f"Error: Output path exists but is not a directory: {models_dir}", file=sys.stderr)
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


    accuracies = {}

    data_file = cfg['dataset']['data_file']

    for model_name, model_class in model_map.items():

        # store per fold accuracies for the model in a dict.
        accuracies[model_name] = {}

        model_dir = os.path.join(models_dir, model_name)

        for r in range(3):
            checkpoint_path = os.path.join(model_dir, f'best-r{r}.ckpt')

            if not os.path.isfile(checkpoint_path):
                print(f"Skipping testing for model {model_name} and rep: {r}, checkpoint not found")
                continue

            kwargs = {
                'in_channels': cfg['dataset']['channels'],
                'num_classes':cfg['dataset']['classes'],
                'loss': nn.CrossEntropyLoss()
            }

            if model_class == models.MESTNet:
                kwargs['wavelet'] = cfg['dataset']['wavelet']
                kwargs['scales']= cfg['dataset']['scales']

            best_model = model_class.load_from_checkpoint(
                checkpoint_path,
                **kwargs
            )

            test_idxs = filter_data_from_h5(
                data_file=data_file,
                sessions = cfg['filter']['sessions'],
                subjects=cfg['filter']['subjects'],
                positions=cfg['filter']['positions'],
                repetitions=r)

            test_dataset = BaseDataset(data_file=data_file, idxs = test_idxs)
            
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size = cfg['validation']['batch_size'],
                shuffle=False,
                num_workers = cfg['validation']['num_workers']
            )
            
            tester = pl.Trainer()

            #results is list, one element per test_loader used, each element is dict of loss+accuracy values
            results = tester.test(best_model, dataloaders = test_loader)
            
            accuracies[model_name][f'r{r}'] = results[0]['test_acc']
    
    # save accuracies for future use
    with open(os.path.join(result_dir, 'eval_results.json'), "w") as outfile:
        json.dump(accuracies, outfile, indent=4)
    
    display_results(accuracy_dict=accuracies)