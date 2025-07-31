from __future__ import annotations

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



def train(model, train_loader, val_loaders, reference_dir, repetition, cfg):
    
    """Runs the initial training phase for a given model."""

    early_stop_callback = EarlyStopping(
        monitor=cfg['early_stop']['monitor'], 
        min_delta=cfg['early_stop']['min_delta'], 
        patience=cfg['early_stop']['patience'], 
        verbose=False, 
        mode=cfg['early_stop']['mode']
    )

    checkpoint_callback = ModelCheckpoint(
      monitor=f"{cfg['early_stop']['monitor']}",
      dirpath=reference_dir,
      filename=f'best-r{repetition}',
      save_top_k=1,
      mode=cfg['early_stop']['mode'],
    )

    print("\nStarting training...")

    trainer = pl.Trainer(
        max_epochs=cfg['training']['epochs'],
        logger=TensorBoardLogger(save_dir=reference_dir),  # Specify the save directory
        callbacks=[early_stop_callback, checkpoint_callback],
        check_val_every_n_epoch=cfg['validation']['frequency'],
        log_every_n_steps=50
    )

    optimizer = torch.optim.Adam(
        model.parameters(), 
        cfg['training']['lr'], 
      )
    model.configure_optimizers = lambda: optimizer

    if cfg['training']['compile'] == True:  
      model = torch.compile(model) # compiles the model and *step (training/validation/prediction)
    
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)
    print("Training finished.")

    return trainer.ckpt_path