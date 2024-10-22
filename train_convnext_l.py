import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from model import CustomConvNeXtLarge
from dataset import CustomImageModule
import config_384_a

if __name__== "__main__":

    data_module = CustomImageModule(config_384_a.TRAIN_DIR, config_384_a.TEST_DIR, config_384_a.SHAPE,
                                    config_384_a.BATCH_SIZE, config_384_a.NUM_WORKERS)

    model = CustomConvNeXtLarge(epochs=config_384_a.MAX_EPOCHS, learning_rate=config_384_a.LEARNING_RATE,
                               scale_factor=config_384_a.SCALE_FACTOR,drop_path_rate=config_384_a.DROP_PATH_RATE,
                               num_classes=config_384_a.NUM_CLASSES, label_smoothing=config_384_a.LABEL_SMOOTHING)
    
    trainer = pl.Trainer(accelerator=config_384_a.ACCELERATOR, devices=config_384_a.DEVICES,
                         precision=config_384_a.PRECISION, max_epochs=config_384_a.MAX_EPOCHS,
                         callbacks=[TQDMProgressBar(leave=True)])

    trainer.fit(model, data_module)

    trainer.test(model, data_module)