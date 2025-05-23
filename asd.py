import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from model import CustomEnsembleModel, CustomModel
from kf_data import CustomImageCSVModule_kf, CustomImageModule_kf
import yaml
import os

def load_hyperparameters(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def set_random_seeds(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_confusion_heatmap(cm, save_path, normalize=True):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Heatmap Normalizado')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    set_random_seeds()
    hyperparams = load_hyperparameters()

    model_base_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}_ne")

    all_labels = []
    all_preds = []

    for fold_idx in range(hyperparams['K_FOLDS']):
        model_filename = f"fold_{fold_idx}_best_model.ckpt"
        model_path = os.path.join(model_base_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"Modelo não encontrado: {model_path}")
            continue

        print(f"Avaliando modelo: {model_path}")
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

        softmax = torch.nn.Softmax(dim=1)

        file_list = data_module.test_ds.image_paths
        fold_labels = []
        fold_preds = []
        all_probs = []

        with torch.no_grad():
            for images, features, labels in test_loader:
                images, features, labels = images.to(device), features.to(device), labels.to(device)
                outputs = model(images, features)
                probs = softmax(outputs)
                preds = torch.argmax(probs, dim=1)

                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        fold_preds = np.array(fold_preds)
        fold_labels = np.array(fold_labels)
        all_probs = np.concatenate(all_probs, axis=0)

        all_preds.extend(fold_preds)
        all_labels.extend(fold_labels)

        acc = accuracy_score(fold_labels, fold_preds)
        print(f"[Fold {fold_idx}] Acurácia: {acc * 100:.2f}%")

        prob_real = all_probs[np.arange(len(fold_labels)), fold_labels]
        prob_pred = all_probs[np.arange(len(fold_labels)), fold_preds]
        discrepancy = prob_pred - prob_real

        errors = fold_preds != fold_labels

        if errors.any():
            idx_discrepancy = np.argmax(discrepancy[errors])
            idx_discrepancy = np.where(errors)[0][idx_discrepancy]

            file_discrepant = file_list[idx_discrepancy]
            true_label_discrepant = fold_labels[idx_discrepancy]
            pred_label_discrepant = fold_preds[idx_discrepancy]

            print(f"[Fold {fold_idx}] Arquivo mais discrepante: {file_discrepant}")
            print(f"[Fold {fold_idx}] Classe real: {true_label_discrepant}")
            print(f"[Fold {fold_idx}] Classe predita: {pred_label_discrepant}")
            print(f"[Fold {fold_idx}] Probabilidade da classe real: {prob_real[idx_discrepancy]:.4f}")
            print(f"[Fold {fold_idx}] Probabilidade da classe predita: {prob_pred[idx_discrepancy]:.4f}")
        else:
            print(f"[Fold {fold_idx}] Nenhum erro encontrado para calcular discrepância.")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    heatmap_filename = f"matriz_confusao_heatmap_todos_folds.png"
    heatmap_path = os.path.join(model_base_dir, heatmap_filename)
    plot_confusion_heatmap(cm, heatmap_path, normalize=True)

if __name__ == '__main__':
    main()
