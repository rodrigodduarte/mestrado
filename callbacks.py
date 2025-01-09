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

class SaveBestOrLastModelCallback(Callback):
    def __init__(self, save_path):
        """
        Callback para salvar o melhor modelo com 100% de val_accuracy e menor val_loss.
        Caso nenhum modelo alcance 100% de val_accuracy, salva o modelo da última época.
        
        Args:
            save_path (str): Caminho para salvar o modelo.
        """
        super().__init__()
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.last_epoch_path = None

    def on_validation_end(self, trainer, pl_module):
        """
        Executado ao final de cada época de validação.
        
        Args:
            trainer (pl.Trainer): Instância do Trainer.
            pl_module (pl.LightningModule): Instância do módulo Lightning.
        """
        # Obtenha as métricas atuais
        metrics = trainer.callback_metrics
        val_accuracy = metrics.get("val_accuracy", 0.0)
        val_loss = metrics.get("val_loss", float('inf'))

        # Salva o modelo da última época
        self.last_epoch_path = f"{self.save_path}_last.ckpt"
        trainer.save_checkpoint(self.last_epoch_path)

        # Verifica se o modelo é melhor com base nas condições
        if val_accuracy == 1.0 and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_accuracy = val_accuracy

            # Salve o melhor modelo
            trainer.save_checkpoint(self.save_path)
            print(f"Novo modelo salvo com 100% val_accuracy e menor val_loss: {val_loss}")

    def on_fit_end(self, trainer, pl_module):
        """
        Executado ao final do treinamento. Caso nenhum modelo atinja 100% de val_accuracy,
        mantém o modelo da última época.
        """
        if self.best_val_accuracy == 1.0:
            print(f"Treinamento finalizado. Melhor modelo salvo com 100% val_accuracy e val_loss: {self.best_val_loss}")
        else:
            print("Treinamento finalizado. Nenhum modelo atingiu 100% val_accuracy.")
            if self.last_epoch_path:
                print(f"Utilizando o modelo da última época salvo em {self.last_epoch_path}.")
                trainer.save_checkpoint(self.save_path)


class EarlyStopOnAccuracyCallback(pl.Callback):
    def __init__(self, target_accuracy: float, max_epoch: int):
        """
        Callback para parar o treinamento caso a métrica val_accuracy não atinja o valor esperado
        até um número máximo de épocas.

        Args:
            target_accuracy (float): O valor mínimo de val_accuracy necessário para continuar o treinamento.
            max_epoch (int): O número máximo de épocas antes de verificar a condição.
        """
        super().__init__()
        self.target_accuracy = target_accuracy
        self.max_epoch = max_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        # Obter a métrica val_accuracy do trainer
        val_accuracy = trainer.callback_metrics.get("val_accuracy")

        # Garantir que estamos na época de interesse e que a métrica existe
        if trainer.current_epoch >= self.max_epoch and val_accuracy is not None:
            if val_accuracy < self.target_accuracy:
                trainer.should_stop = True
                print(f"Stopping early: val_accuracy {val_accuracy:.4f} did not reach {self.target_accuracy} by epoch {self.max_epoch}.")