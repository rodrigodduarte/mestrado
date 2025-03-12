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
        Callback para salvar o melhor modelo com base no menor val_loss.
        Caso não seja possível obter val_loss, salva o modelo da última época
        ao final do treinamento.

        Args:
            save_path (str): Caminho base para salvar o modelo (best).
        """
        super().__init__()
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.best_epoch = None  # Armazenará a época em que obtemos o melhor val_loss
        self.last_epoch_path = None

    def on_validation_end(self, trainer, pl_module):
        """
        Executado ao final de cada época de validação.
        """
        # Obtenha as métricas atuais
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss", float('inf'))

        # Caminho para o modelo da última época
        self.last_epoch_path = f"{self.save_path}_last.ckpt"

        # Sempre salva o checkpoint da última época
        trainer.save_checkpoint(self.last_epoch_path)

        # Verifica se o val_loss atual é o melhor até agora
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = trainer.current_epoch  # Salva a época do melhor modelo
            # Salva o melhor modelo com base no val_loss
            trainer.save_checkpoint(self.save_path)
            # Removemos o print aqui para não ficar imprimindo a cada época

    def on_fit_end(self, trainer, pl_module):
        """
        Executado ao final do treinamento. Caso não tenha sido registrado
        nenhum val_loss (mantido em 'inf'), utiliza o modelo da última época.
        """
        if self.best_val_loss == float('inf'):
            # Significa que nunca encontramos um val_loss válido
            print("Treinamento finalizado. Não foi possível obter val_loss válido.")
            if self.last_epoch_path:
                print(f"Utilizando o modelo da última época salvo em {self.last_epoch_path}.")
                # Copia o modelo da última época para o 'melhor' caminho:
                trainer.save_checkpoint(self.save_path)
        else:
            print(f"Treinamento finalizado. Melhor val_loss encontrado: {self.best_val_loss:.4f}")
            print(f"Modelo da época {self.best_epoch} salvo em {self.save_path}.")


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

class EarlyStopCallback(pl.callbacks.Callback):
    def __init__(self, metric_name: str, threshold: float, target_epoch: int):
        """
        Callback para interromper o treinamento com base em uma métrica, limite e época específicos.

        Args:
            metric_name (str): Nome da métrica a ser monitorada (e.g., 'val_loss').
            threshold (float): Valor limite da métrica.
            target_epoch (int): Época na qual verificar a métrica (índice começa em 0).
        """
        super().__init__()
        self.metric_name = metric_name
        self.threshold = threshold
        self.target_epoch = target_epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        # Verifica se estamos na época alvo
        if trainer.current_epoch == self.target_epoch:
            # Obtém o valor da métrica monitorada
            metric_value = trainer.callback_metrics.get(self.metric_name)

            if metric_value is None:
                print(f"Aviso: '{self.metric_name}' não encontrado nos callback_metrics.")
                return

            # Interrompe o treinamento se o valor da métrica exceder o limite
            if metric_value >= self.threshold:
                print(
                    f"Interrompendo o treinamento na época {trainer.current_epoch + 1}. "
                    f"{self.metric_name} ({metric_value:.4f}) >= {self.threshold}."
                )
                trainer.should_stop = True
                

class StopAllFoldsCallback(Callback):
    def __init__(self, patience=1, threshold=1e-3, monitor="val_loss", mode="min", verbose=False):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_metric = None
        self.counter = 0
        self.stop_training = False

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stop_training:
            trainer.should_stop = True
            return

        current_metric = trainer.callback_metrics.get(self.monitor)
        if current_metric is None:
            return

        if self.best_metric is None:
            self.best_metric = current_metric
            return

        if self.mode == "min":
            improvement = self.best_metric - current_metric
        else:
            improvement = current_metric - self.best_metric

        if improvement > self.threshold:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"🚨 Stop All Folds acionado! {self.monitor} não melhorou por {self.patience} épocas. Cancelando todos os folds e iniciando nova run.")
                self.stop_training = True
                trainer.should_stop = True
                raise SystemExit("Treinamento interrompido para iniciar nova run.")
