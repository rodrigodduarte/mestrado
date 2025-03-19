import torch
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

# Inicializar o trainer do PyTorch Lightning
trainer = pl.Trainer(accelerator='auto', devices=1)

# Avaliação do modelo
results = trainer.test(model, datamodule=data_module)

# Extrair métricas
acc_value = results[0]['test_accuracy']
prec_value = results[0]['test_precision']
rec_value = results[0]['test_recall']
f1_value = results[0]['test_f1']
conf_matrix_value = results[0]['test_confusion_matrix'].cpu().numpy()

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
conf_matrix_path = os.path.join(conf_matrix_dir, f"mc_{os.path.basename(hyperparams['CM_PATH'])}_{hyperparams["RUN_NAME"]}.png")
plot_confusion_matrix(conf_matrix_value, save_path=conf_matrix_path)
