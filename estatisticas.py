import torch
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
model.eval()

# Criar diretório para salvar as matrizes de confusão
conf_matrix_dir = os.path.join("confusion_matrix")
os.makedirs(conf_matrix_dir, exist_ok=True)
exemplos_dir = os.path.join(conf_matrix_dir, "exemplos")
os.makedirs(exemplos_dir, exist_ok=True)

# Configurar o DataLoader de Teste
data_module = CustomImageModule_kf(
    train_dir=hyperparams['TRAIN_DIR'],
    test_dir=hyperparams['TEST_DIR'],
    shape=hyperparams['SHAPE'],
    batch_size=hyperparams['BATCH_SIZE'],
    num_workers=hyperparams['NUM_WORKERS']
)
data_module.setup(stage='test')
test_loader = data_module.test_dataloader()

# Garantir que o modelo está na GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Avaliação do modelo
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(images.cpu())

# Converter para tensores
all_preds = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

# Identificar exemplos classificados incorretamente
incorrect_indices = (all_preds != all_labels).nonzero(as_tuple=True)[0].tolist()
incorrect_examples = [(all_images[i], all_labels[i].item(), all_preds[i].item()) for i in incorrect_indices]

# Selecionar no máximo 3 exemplos incorretos
incorrect_examples = incorrect_examples[:3] if len(incorrect_examples) > 3 else incorrect_examples

# Exibir resultados
print(f"Total de exemplos classificados incorretamente: {len(incorrect_indices)}")

for i, (image, true_label, pred_label) in enumerate(incorrect_examples):
    plt.figure(figsize=(10, 5))
    
    # Plot da imagem incorretamente classificada
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Errado: {pred_label}, Correto: {true_label}")
    plt.axis("off")
    
    # Selecionar um exemplo aleatório da classe predita para comparação
    class_examples = [img for img, label, _ in incorrect_examples if label == pred_label]
    if class_examples:
        reference_image = random.choice(class_examples)
        plt.subplot(1, 2, 2)
        plt.imshow(reference_image.permute(1, 2, 0))
        plt.title(f"Exemplo da Classe {pred_label}")
        plt.axis("off")
    
    # Salvar a imagem incorretamente classificada
    erro_path = os.path.join(exemplos_dir, f"exemplo_errado_{i}.png")
    plt.savefig(erro_path)
    print(f"Imagem salva em: {erro_path}")
    
    plt.show()

# Inicializar métricas
num_classes = len(torch.unique(all_labels))
accuracy = Accuracy(task='multiclass', num_classes=num_classes)
precision = Precision(task='multiclass', num_classes=num_classes)
recall = Recall(task='multiclass', num_classes=num_classes)
f1 = F1Score(task='multiclass', num_classes=num_classes)
conf_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)

# Calcular métricas
acc_value = accuracy(all_preds, all_labels).item()
prec_value = precision(all_preds, all_labels).item()
rec_value = recall(all_preds, all_labels).item()
f1_value = f1(all_preds, all_labels).item()
conf_matrix_value = conf_matrix(all_preds, all_labels).cpu().numpy()

# Exibir resultados
print(f"Acurácia: {acc_value:.4f}")
print(f"Precisão: {prec_value:.4f}")
print(f"Recall: {rec_value:.4f}")
print(f"F1-Score: {f1_value:.4f}")

# Exibir e salvar matriz de confusão
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.savefig(save_path)
    plt.show()

# Criar nome do arquivo usando hyperparams["CM_PATH"]
conf_matrix_path = os.path.join(conf_matrix_dir, f"mc_{os.path.basename(hyperparams['CM_PATH'])}_{hyperparams['TMODEL']}_{hyperparams["RUN_NAME"]}.png")
plot_confusion_matrix(conf_matrix_value, save_path=conf_matrix_path)
