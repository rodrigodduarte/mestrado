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

# Obter previsões e rótulos do modelo
all_preds = model.test_preds.cpu().numpy()
all_labels = model.test_labels.cpu().numpy()
all_image_paths = model.test_image_paths

# Identificar exemplos classificados incorretamente
incorrect_indices = np.where(all_preds != all_labels)[0]
incorrect_examples = [(all_image_paths[i], all_labels[i], all_preds[i]) for i in incorrect_indices]

# Selecionar no máximo 3 exemplos incorretos
incorrect_examples = incorrect_examples[:3] if len(incorrect_examples) > 3 else incorrect_examples

for i, (image_path, true_label, pred_label) in enumerate(incorrect_examples):
    plt.figure(figsize=(10, 5))
    
    # Carregar a imagem original do dataset
    image = Image.open(image_path)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Errado: {pred_label}, Correto: {true_label}")
    plt.axis("off")
    
    # Selecionar uma imagem correta da classe predita
    class_samples = [p for p, l in zip(all_image_paths, all_labels) if l == pred_label]
    reference_image_path = random.choice(class_samples) if class_samples else None

    if reference_image_path:
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