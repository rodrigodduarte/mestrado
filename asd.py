import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf
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

def plot_confusion_heatmap(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    set_random_seeds()
    hyperparams = load_hyperparameters()

    fold_idx = 0  # Escolha o fold desejado

    model_base_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}")
    model_filename = f"fold_{fold_idx}_best_model.ckpt"
    model_path = os.path.join(model_base_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"Modelo não encontrado: {model_path}")
        return

    print(f"Avaliando modelo: {model_path}")
    model = CustomEnsembleModel.load_from_checkpoint(model_path)
    model.eval()

    data_module = CustomImageCSVModule_kf(
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
    all_probs = []

    test_dataset = data_module.test_ds
    file_list = test_dataset.image_paths

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, features, labels in test_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features)
            probs = softmax(outputs)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    errors = all_preds != all_labels

    if errors.any():
        idx_error = np.where(errors)[0][0]
        true_label = all_labels[idx_error]
        pred_label = all_preds[idx_error]

        file_error = file_list[idx_error]

        print(f"Arquivo mal classificado: {file_error}")
        print(f"Classe real: {true_label}, foi classificado como: {pred_label}")
    else:
        print("Nenhum erro encontrado.")

    # Encontrar a imagem com maior discrepância
    prob_real = all_probs[np.arange(len(all_labels)), all_labels]
    prob_pred = all_probs[np.arange(len(all_labels)), all_preds]
    discrepancy = prob_pred - prob_real

    idx_discrepancy = np.argmax(discrepancy)

    file_discrepant = file_list[idx_discrepancy]
    true_label_discrepant = all_labels[idx_discrepancy]
    pred_label_discrepant = all_preds[idx_discrepancy]

    print(f"\nArquivo mais discrepante: {file_discrepant}")
    print(f"Classe real: {true_label_discrepant}")
    print(f"Classe predita: {pred_label_discrepant}")
    print(f"Probabilidade da classe real: {prob_real[idx_discrepancy]:.4f}")
    print(f"Probabilidade da classe predita: {prob_pred[idx_discrepancy]:.4f}")

    # Gerar e salvar heatmap da matriz de confusão
    heatmap_filename = f"fold_{fold_idx}_matriz_confusao_heatmap.png"
    heatmap_path = os.path.join(model_base_dir, heatmap_filename)
    plot_confusion_heatmap(all_labels, all_preds, heatmap_path)

if __name__ == '__main__':
    main()