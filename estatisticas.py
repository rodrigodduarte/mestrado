import torch
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from model import CustomModel
from kf_data import CustomImageModule_kf
import yaml
import os

# Carregar hiperparâmetros
def load_hyperparameters(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Definir semente fixa para reproduzir resultados
set_random_seeds()

# Carregar modelo final
hyperparams = load_hyperparameters()
final_model_path = os.path.join(f'{hyperparams['NAME_DATASET']}_bestmodel', hyperparams["RUN_NAME"], 'best_model.ckpt')

print(f"Carregando modelo final de: {final_model_path}")
model = CustomModel.load_from_checkpoint(final_model_path)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Criar diretório para salvar as matrizes de confusão e exemplos
conf_matrix_dir = os.path.join("confusion_matrix")
os.makedirs(conf_matrix_dir, exist_ok=True)
exemplos_dir = os.path.join(conf_matrix_dir, "exemplos")
os.makedirs(exemplos_dir, exist_ok=True)


# Configurar o DataModule de Teste
data_module = CustomImageModule_kf(
    train_dir=hyperparams['TRAIN_DIR'],
    test_dir=hyperparams['TEST_DIR'],
    shape=hyperparams['SHAPE'],
    batch_size=hyperparams['BATCH_SIZE'],
    num_workers=hyperparams['NUM_WORKERS']
)
data_module.setup(stage='test')

# Usar o Trainer para rodar o teste
trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
results = trainer.test(model, datamodule=data_module)
