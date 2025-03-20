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

# Calcular a matriz de confusão a partir do modelo
conf_matrix_value = model.on_test_epoch_end()

# Exibir e salvar a matriz de confusão
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

# Identificar imagens classificadas incorretamente
incorrect_samples = model.get_misclassified_samples()
incorrect_samples = incorrect_samples[:3] if len(incorrect_samples) > 3 else incorrect_samples

for i, (image_path, true_label, pred_label) in enumerate(incorrect_samples):
    plt.figure(figsize=(10, 5))
    
    # Carregar a imagem original do dataset
    image = Image.open(image_path)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Errado: {pred_label}, Correto: {true_label}")
    plt.axis("off")
    
    # Selecionar uma imagem do dataset da classe predita para comparação
    reference_image_path = model.get_sample_from_class(pred_label)
    reference_image = Image.open(reference_image_path)
    plt.subplot(1, 2, 2)
    plt.imshow(reference_image)
    plt.title(f"Exemplo da Classe {pred_label}")
    plt.axis("off")
    
    # Salvar a imagem no diretório de exemplos
    erro_path = os.path.join(exemplos_dir, os.path.basename(image_path))
    image.save(erro_path)
    print(f"Imagem salva em: {erro_path}")
    
    plt.show()
