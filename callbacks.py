import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import time
import wandb

class CustomEarlyStopping(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics['val_accuracy']

class ImagesPerSecondCallback(pl.Callback):
    def __init__(self):
        self.start_time = None
        self.total_images = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # Inicia o temporizador no início de cada época
        self.start_time = time.time()
        self.total_images = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Atualiza o contador de imagens processadas
        batch_size = batch[0].size(0)  # Supondo que `batch[0]` são as imagens
        self.total_images += batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        # Calcula o tempo total e as imagens por segundo ao final de cada época
        elapsed_time = time.time() - self.start_time
        images_per_second = self.total_images / elapsed_time

        # Loga a métrica diretamente no W&B
        wandb.log({"images_per_second": images_per_second, "epoch": trainer.current_epoch})

import pytorch_lightning as pl

class EarlyStoppingAtSpecificEpoch(pl.Callback):
    def __init__(self, stop_epoch: int, threshold: float):
        """
        Callback para interromper o treinamento se o val_loss for maior que o limiar
        em uma época específica.

        Args:
            stop_epoch (int): A época específica em que o `val_loss` será verificado.
            threshold (float): Valor limiar para o val_loss.
        """
        super().__init__()
        self.stop_epoch = stop_epoch
        self.threshold = threshold

    def on_validation_epoch_end(self, trainer, pl_module):
        # Obtém o valor do val_loss
        val_loss = trainer.callback_metrics.get("val_loss")

        # Verifica se o val_loss foi registrado e se estamos na época correta
        if val_loss is None:
            return

        # Interrompe o treinamento apenas se estivermos na época específica
        if trainer.current_epoch == self.stop_epoch:
            if val_loss > self.threshold:
                print(f"Early stopping: val_loss {val_loss:.4f} maior que {self.threshold} na época {self.stop_epoch}.")
                trainer.should_stop = True