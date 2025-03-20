import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from model import CustomModel
from kf_data import CustomImageModule_kf
import yaml
import os

# Carregar hiperparâmetros
def load_hyperparameters(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Definir semente fixa para reproduzir resultados
def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seeds()

# Carregar modelo final
hyperparams = load_hyperparameters()
final_model_path = os.path.join(f'{hyperparams['NAME_DATASET']}_bestmodel', hyperparams["RUN_NAME"], 'best_model.ckpt')

print(f"Carregando modelo final de: {final_model_path}")
model = CustomModel.load_from_checkpoint(final_model_path)
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Criar diretório para salvar a matriz de confusão
conf_matrix_dir = os.path.join("confusion_matrix")
os.makedirs(conf_matrix_dir, exist_ok=True)

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
trainer.test(model, datamodule=data_module)

# Calcular matriz de confusão
conf_matrix_value = model.on_test_epoch_end()

# Salvar a matriz de confusão
conf_matrix_path = os.path.join(conf_matrix_dir, f"mc_{hyperparams['RUN_NAME']}.png")
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig(save_path)
    plt.show()

plot_confusion_matrix(conf_matrix_value, save_path=conf_matrix_path)
print(f"✅ Matriz de Confusão salva em: {conf_matrix_path}")
