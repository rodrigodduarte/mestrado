import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler

from model import CustomConvNeXtTiny, CustomConvNeXtLarge, CustomCCT_14_7x2_224, CustomCCT_14_7x2_384
from dataset import CustomImageModule
import config

import wandb
from pytorch_lightning.loggers import WandbLogger

import random


if __name__== "__main__":
    wandb.login()

    hyperparameters = dict(
    shape = config.SHAPE,
    epochs = config.MAX_EPOCHS,
    classes = config.NUM_CLASSES,
    batch_size = config.BATCH_SIZE,
    learning_rate = config.LEARNING_RATE,
    weight_decay = config.WEIGHT_DECAY,
    optimizer_momentum = config.OPTIMIZER_MOMENTUM,
    scale_factor = config.SCALE_FACTOR,
    drop_path_rate = config.DROP_PATH_RATE,
    label_smoothing = config.LABEL_SMOOTHING,
    num_workers = config.NUM_WORKERS,
    precision = config.PRECISION
)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    data_module = CustomImageModule(config.TRAIN_DIR, config.TEST_DIR, config.SHAPE,
                                    config.BATCH_SIZE, config.NUM_WORKERS)

    with wandb.init(project="swedish calssification with lightning", config=hyperparameters):
        wandb_logger = WandbLogger(project="CCT_14_7x2_224")

        model = CustomCCT_14_7x2_384(
            epochs=config.MAX_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            scale_factor=config.SCALE_FACTOR,
            drop_path_rate=config.DROP_PATH_RATE,
            num_classes=config.NUM_CLASSES,
            label_smoothing=config.LABEL_SMOOTHING)
        
        trainer = pl.Trainer(
            logger=wandb_logger,    # W&B integration
            log_every_n_steps=50,   # set the logging frequency
            profiler="simple",
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            precision=config.PRECISION,
            max_epochs=config.MAX_EPOCHS,
            callbacks=[TQDMProgressBar(leave=True)])

        trainer.fit(model, data_module)

        trainer.test(model, data_module)

        wandb.finish()