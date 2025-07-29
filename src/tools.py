from __future__ import annotations

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import models


def train(model, train_loader, val_loaders, reference_dir, repetition, cfg):
    
    """Runs the initial training phase for a given model."""

    early_stop_callback = EarlyStopping(
        monitor=f"{cfg['early_stop']['monitor']}/dataloader_idx_0", 
        min_delta=cfg['early_stop']['min_delta'], 
        patience=cfg['early_stop']['patience'], 
        verbose=False, 
        mode=cfg['early_stop']['mode']
    )

    checkpoint_callback_initial = ModelCheckpoint(
      monitor=f"{cfg['early_stop']['monitor']}/dataloader_idx_0",
      dirpath=reference_dir,
      filename=f'inter{repetition}',
      save_top_k=1,
      mode=cfg['early_stop']['mode'],
    )

    print("\nStarting initial training...")

    trainer_initial = pl.Trainer(
        max_epochs=cfg['training']['epochs'],
        logger=TensorBoardLogger(save_dir=reference_dir),  # Specify the save directory
        callbacks=[early_stop_callback, checkpoint_callback_initial],
        check_val_every_n_epoch=2,
        log_every_n_steps=50
    )

    optimizer = torch.optim.Adam(
        model.parameters(), 
        cfg['training']['lr'], 
      )
    model.configure_optimizers = lambda: optimizer

    if cfg['training']['compile'] == True:  
      model = torch.compile(model) # compiles the model and *step (training/validation/prediction)
    
    trainer_initial.fit(model, train_loader, val_dataloaders=val_loaders)
    print("Initial training finished.")

    best_checkpoint_path = checkpoint_callback_initial.best_model_path
    print(f"Loading best model from: {best_checkpoint_path}")

    kwargs = {
       'in_channels': cfg['dataset']['channels'],
       'num_classes': cfg['dataset']['num_classes'],
       'loss': torch.nn.CrossEntropyLoss()
       }
    
    if model.__class__ == models.MESTNet:
       kwargs['wavelet'] = cfg['dataset']['wavelet']
       kwargs['scales'] = cfg['dataset']['scales']

    # Load the best model state
    model = model.__class__.load_from_checkpoint(
        best_checkpoint_path,
        **kwargs
    )

    # Update the optimizer for the finetuning model
    finetune_optimizer = torch.optim.Adam(
        model.parameters(), 
        cfg['finetuning']['lr'], 
      )
    
    model.configure_optimizers = lambda: finetune_optimizer

    checkpoint_callback_finetune = ModelCheckpoint(
      monitor='train_loss',
      dirpath=reference_dir,
      filename=f'r{repetition}_final',
      save_top_k=1,
      mode='min',
    )

    trainer_finetune = pl.Trainer(
        max_epochs=cfg['finetuning']['epochs'],
        logger=TensorBoardLogger(save_dir=reference_dir),  # Specify the save directory
        callbacks=[checkpoint_callback_finetune]
    )

    combined_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, val_loaders[0].dataset])
    combined_loader = torch.utils.data.DataLoader(
       combined_dataset, 
       batch_size=cfg['training']['batch_size'], 
       shuffle=True, 
       num_workers=cfg['training']['num_workers']
    )

    print("\nStarting finetuning...")

    #compiled_model = torch.compile(model)
    trainer_finetune.fit(model, combined_loader)
    
    return trainer_finetune.ckpt_path