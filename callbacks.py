import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class CustomEarlyStopping(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics['val_accuracy']

        # Verifica se a acurácia de validação é 1
        if current_score == 1.0:
            print("Acurácia de validação atingiu 1.0! Interrompendo treinamento.")
            trainer.should_stop = True