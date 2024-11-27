import torch
import pytorch_lightning as pl
import yaml
import wandb
from model import CustomEnsembleModel
from dataset import CustomImageCSVModule

# Função para carregar hiperparâmetros do arquivo YAML
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)  # Carregar o YAML
    return hyperparams

def infer_model():
    hyperparams = load_hyperparameters('config.yaml')

    # Definir o módulo de dados com os parâmetros do arquivo de configuração
    data_module = CustomImageCSVModule(
        train_dir=hyperparams['TRAIN_DIR'],
        test_dir=hyperparams['TEST_DIR'],
        shape=hyperparams['SHAPE'],
        batch_size=hyperparams['BATCH_SIZE'],
        num_workers=hyperparams['NUM_WORKERS']
    )

    # Caminho para o checkpoint salvo
    best_model_path = "/home/rodrigoduarte/Documentos/projeto/checkpoints_config/epochepoch=93-val_lossval_loss=0.66.ckpt"

    # Carregar o modelo a partir do checkpoint
    model = CustomEnsembleModel.load_from_checkpoint(best_model_path)

    # Configurar o Trainer do PyTorch Lightning para teste
    trainer = pl.Trainer(
        accelerator=hyperparams['ACCELERATOR'],
        devices=hyperparams['DEVICES'],
        precision=hyperparams['PRECISION']
    )

    # Realizar inferência/teste no conjunto de teste
    trainer.test(model, data_module)

if __name__ == "__main__":
    # Login no WandB (se necessário)
    wandb.login()

    # Realizar inferência no conjunto de teste com o modelo carregado
    infer_model()
