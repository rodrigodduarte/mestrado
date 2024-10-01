import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from model import CustomConvNeXtLarge
from dataset import CustomImageModule
import config_convnext_l

if __name__== "__main__":

    data_module = CustomImageModule(config_convnext_l.TRAIN_DIR, config_convnext_l.TEST_DIR, config_convnext_l.SHAPE,
                                    config_convnext_l.BATCH_SIZE, config_convnext_l.NUM_WORKERS)

    model = CustomConvNeXtLarge(epochs=config_convnext_l.MAX_EPOCHS, learning_rate=config_convnext_l.LEARNING_RATE,
                               scale_factor=config_convnext_l.SCALE_FACTOR,drop_path_rate=config_convnext_l.DROP_PATH_RATE,
                               num_classes=config_convnext_l.NUM_CLASSES, label_smoothing=config_convnext_l.LABEL_SMOOTHING)
    
    trainer = pl.Trainer(accelerator=config_convnext_l.ACCELERATOR, devices=config_convnext_l.DEVICES,
                         precision=config_convnext_l.PRECISION, max_epochs=config_convnext_l.MAX_EPOCHS,
                         callbacks=[TQDMProgressBar(leave=True)])

    trainer.fit(model, data_module)

    trainer.test(model, data_module)