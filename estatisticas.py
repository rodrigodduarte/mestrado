import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf
import yaml
import os

# Carregar hiperparâmetros
def load_hyperparameters(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Carregar modelo final
hyperparams = load_hyperparameters()
final_model_path = os.path.join('flavia_bestmodel', 'runs', hyperparams["RUN_NAME"], 'best_model.ckpt')

print(f"Carregando modelo final de: {final_model_path}")
model = CustomEnsembleModel.load_from_checkpoint(final_model_path)
model.eval()

# Criar diretório para salvar as matrizes de confusão
conf_matrix_dir = os.path.join("confusion_matrix")
os.makedirs(conf_matrix_dir, exist_ok=True)

# Configurar o DataLoader de Teste
data_module = CustomImageCSVModule_kf(
    train_dir=hyperparams['TRAIN_DIR'],
    test_dir=hyperparams['TEST_DIR'],
    shape=hyperparams['SHAPE'],
    batch_size=hyperparams['BATCH_SIZE'],
    num_workers=hyperparams['NUM_WORKERS'],
    n_splits=hyperparams['K_FOLDS'],
    fold_idx=0
)
data_module.setup(stage='test')
test_loader = data_module.test_dataloader()

# Avaliação do modelo
all_preds = []
all_labels = []
with torch.no_grad():
    for images, features, labels in test_loader:
        outputs = model(images, features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Converter para tensores
all_preds = torch.tensor(all_preds)
all_labels = torch.tensor(all_labels)

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
conf_matrix_path = os.path.join(conf_matrix_dir, f"mc_{hyperparams['CM_PATH']}.png")
plot_confusion_matrix(conf_matrix_value, save_path=conf_matrix_path)
