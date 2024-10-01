import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from model import CustomConvNeXtTiny
from dataset import CustomImageModule
import config

if __name__== "__main__":

    data_module = CustomImageModule(config.TRAIN_DIR, config.TEST_DIR, config.SHAPE,
                                    config.BATCH_SIZE, config.NUM_WORKERS)

    model = CustomConvNeXtTiny(epochs=config.MAX_EPOCHS, learning_rate=config.LEARNING_RATE,
                               scale_factor=config.SCALE_FACTOR,drop_path_rate=config.DROP_PATH_RATE,
                               num_classes=config.NUM_CLASSES, label_smoothing=config.LABEL_SMOOTHING)
    
    trainer = pl.Trainer(accelerator=config.ACCELERATOR, devices=config.DEVICES,
                         precision=config.PRECISION, max_epochs=config.MAX_EPOCHS,
                         callbacks=[TQDMProgressBar(leave=True)])

    trainer.fit(model, data_module)

    trainer.test(model, data_module)