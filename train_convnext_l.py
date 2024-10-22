import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from model import CustomConvNeXtLarge
from dataset import CustomImageModule
import config_384

if __name__== "__main__":

    data_module = CustomImageModule(config_384.TRAIN_DIR, config_384.TEST_DIR, config_384.SHAPE,
                                    config_384.BATCH_SIZE, config_384.NUM_WORKERS)

    model = CustomConvNeXtLarge(epochs=config_384.MAX_EPOCHS, learning_rate=config_384.LEARNING_RATE,
                               scale_factor=config_384.SCALE_FACTOR,drop_path_rate=config_384.DROP_PATH_RATE,
                               num_classes=config_384.NUM_CLASSES, label_smoothing=config_384.LABEL_SMOOTHING)
    
    trainer = pl.Trainer(accelerator=config_384.ACCELERATOR, devices=config_384.DEVICES,
                         precision=config_384.PRECISION, max_epochs=config_384.MAX_EPOCHS,
                         callbacks=[TQDMProgressBar(leave=True)])

    trainer.fit(model, data_module)

    trainer.test(model, data_module)