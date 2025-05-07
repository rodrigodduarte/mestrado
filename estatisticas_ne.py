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

def plot_confusion_matrix(cm, save_path, title='Matriz de Confusão'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

set_random_seeds()
hyperparams = load_hyperparameters()

# Diretório padrão onde os modelos foram salvos durante o treinamento
model_base_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}_ne")

# Avaliar cada modelo de cada fold
for fold_idx in range(hyperparams['K_FOLDS']):
    model_filename = f"fold_{fold_idx}_best_model.ckpt"
    model_path = os.path.join(model_base_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"[Fold {fold_idx}] Modelo não encontrado: {model_path}. Pulando.")
        continue

    print(f"[Fold {fold_idx}] Avaliando modelo: {model_path}")
    model = CustomModel.load_from_checkpoint(model_path)
    model.eval()

    data_module = CustomImageModule_kf(
        train_dir=hyperparams['TRAIN_DIR'],
        test_dir=hyperparams['TEST_DIR'],
        shape=hyperparams['SHAPE'],
        batch_size=hyperparams['BATCH_SIZE'],
        num_workers=hyperparams['NUM_WORKERS'],
        n_splits=hyperparams['K_FOLDS'],
        fold_idx=fold_idx
    )
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    num_classes = len(torch.unique(all_labels))
    accuracy = Accuracy(task='multiclass', num_classes=num_classes)
    precision = Precision(task='multiclass', num_classes=num_classes)
    recall = Recall(task='multiclass', num_classes=num_classes)
    f1 = F1Score(task='multiclass', num_classes=num_classes)
    conf_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)

    acc_value = accuracy(all_preds, all_labels).item()
    prec_value = precision(all_preds, all_labels).item()
    rec_value = recall(all_preds, all_labels).item()
    f1_value = f1(all_preds, all_labels).item()
    conf_matrix_value = conf_matrix(all_preds, all_labels).cpu().numpy()

    print(f"[Fold {fold_idx}] Acurácia: {acc_value:.4f} | Precisão: {prec_value:.4f} | Recall: {rec_value:.4f} | F1: {f1_value:.4f}")

    # Salvar a matriz de confusão no mesmo diretório do modelo
    matrix_filename = model_filename.replace(".ckpt", ".png")
    matrix_path = os.path.join(model_base_dir, matrix_filename)
    plot_confusion_matrix(conf_matrix_value, save_path=matrix_path, title=f"Matriz de Confusão - Fold {fold_idx}")
