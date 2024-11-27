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

class EarlyStoppingAtSpecificEpoch(Callback):
    def __init__(self, patience=3, threshold=1e-3, monitor="val_loss", mode="min", verbose=False):
        """
        Interrompe o treinamento se a métrica monitorada não melhorar após um número de épocas.

        :param patience: Número de épocas sem melhoria após as quais o treinamento será interrompido.
        :param threshold: A diferença mínima que deve ser observada na métrica para ser considerada uma melhoria.
        :param monitor: A métrica a ser monitorada (por exemplo, 'val_loss', 'val_accuracy').
        :param mode: O modo da métrica ('min' ou 'max'). Se 'min', menor valor é melhor, se 'max', maior valor é melhor.
        :param verbose: Se True, imprime mensagens de log quando o treinamento for interrompido.
        """
        self.patience = patience
        self.threshold = threshold
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_metric = None
        self.counter = 0

    def on_epoch_end(self, trainer, pl_module):
        """
        A cada fim de época, verifica a melhoria na métrica monitorada.
        """
        current_metric = trainer.callback_metrics.get(self.monitor)

        if current_metric is None:
            return

        # Se a métrica monitorada não tiver um valor anterior, inicializamos
        if self.best_metric is None:
            self.best_metric = current_metric
            return

        # Verifica se a melhoria é suficiente
        if self.mode == "min":
            improvement = self.best_metric - current_metric
        elif self.mode == "max":
            improvement = current_metric - self.best_metric

        if improvement > self.threshold:
            # Se a melhoria for maior que o threshold, atualiza a melhor métrica e reseta o contador
            self.best_metric = current_metric
            self.counter = 0
        else:
            # Caso contrário, incrementa o contador
            self.counter += 1

            # Se o contador ultrapassar a paciência, interrompe o treinamento
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Treinamento interrompido antecipadamente devido à falta de melhoria em {self.monitor}.")
                trainer.should_stop = True