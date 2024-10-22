import torch
import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.profilers import PyTorchProfiler

from model import CustomConvNeXtTiny, CustomConvNeXtSmall, CustomConvNeXtBase, CustomConvNeXtLarge
from dataset import CustomImageModule
import config_384_b

import wandb
from pytorch_lightning.loggers import WandbLogger

import random


if __name__== "__main__":
    wandb.login()

    hyperparameters = dict(
    shape = config_384_b.SHAPE,
    epochs = config_384_b.MAX_EPOCHS,
    classes = config_384_b.NUM_CLASSES,
    batch_size = config_384_b.BATCH_SIZE,
    learning_rate = config_384_b.LEARNING_RATE,
    weight_decay = config_384_b.WEIGHT_DECAY,
    optimizer_momentum = config_384_b.OPTIMIZER_MOMENTUM,
    scale_factor = config_384_b.SCALE_FACTOR,
    drop_path_rate = config_384_b.DROP_PATH_RATE,
    label_smoothing = config_384_b.LABEL_SMOOTHING,
    num_workers = config_384_b.NUM_WORKERS,
    precision = config_384_b.PRECISION
)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
    data_module = CustomImageModule(config_384_b.TRAIN_DIR, config_384_b.TEST_DIR, config_384_b.SHAPE,
                                    config_384_b.BATCH_SIZE, config_384_b.NUM_WORKERS)

    with wandb.init(project="swedish calssification with lightning", config=hyperparameters):
        wandb_logger = WandbLogger(project="swedish calssification with lightning")

        model = CustomConvNeXtSmall(
            epochs=config_384_b.MAX_EPOCHS,
            learning_rate=config_384_b.LEARNING_RATE,
            scale_factor=config_384_b.SCALE_FACTOR,
            drop_path_rate=config_384_b.DROP_PATH_RATE,
            num_classes=config_384_b.NUM_CLASSES,
            label_smoothing=config_384_b.LABEL_SMOOTHING)
        
        trainer = pl.Trainer(
            logger=wandb_logger,    # W&B integration
            log_every_n_steps=50,   # set the logging frequency
            profiler="simple",
            accelerator=config_384_b.ACCELERATOR,
            devices=config_384_b.DEVICES,
            precision=config_384_b.PRECISION,
            max_epochs=config_384_b.MAX_EPOCHS,
            callbacks=[TQDMProgressBar(leave=True)])

        trainer.fit(model, data_module)

        trainer.test(model, data_module)

        wandb.finish()