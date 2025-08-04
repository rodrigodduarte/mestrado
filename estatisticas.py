import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf
import yaml
import os

# --- NOVOS IMPORTS -----------------------------------------------------------
from scipy import stats                # intervalo de confiança (t-Student)
from datetime import datetime          # carimbo de data/hora
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------#
# Utilidades
# -----------------------------------------------------------------------------#
def load_hyperparameters(file_path: str = "config.yaml"):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def set_random_seeds(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(cm, save_path, title: str = "Matriz de Confusão"):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def print_final_stats(metric_list, name):
    """Exibe e devolve (média, desvio) de uma métrica sobre os folds."""
    metric_array = np.array(metric_list)
    print(f"{name} por Fold: {metric_array}")
    print(
        f"{name} Média: {metric_array.mean():.4f} | "
        f"Desvio Padrão: {metric_array.std():.4f}\n"
    )
    return metric_array.mean(), metric_array.std()


# -----------------------------------------------------------------------------#
# Configuração inicial
# -----------------------------------------------------------------------------#
set_random_seeds()
hyperparams = load_hyperparameters()

model_base_dir = os.path.join(
    "modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}"
)

# Listas para armazenar as métricas de todos os folds
acc_list, prec_list, rec_list, f1_list, loss_list = ([] for _ in range(5))
fold_metrics = {}

# -----------------------------------------------------------------------------#
# Avaliação k-fold
# -----------------------------------------------------------------------------#
for fold_idx in range(hyperparams["K_FOLDS"]):
    model_filename = f"fold_{fold_idx}_best_model.ckpt"
    model_path = os.path.join(model_base_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"[Fold {fold_idx}] Modelo não encontrado: {model_path}. Pulando.")
        continue

    print(f"[Fold {fold_idx}] Avaliando modelo: {model_path}")
    model = CustomEnsembleModel.load_from_checkpoint(model_path)
    model.eval()

    data_module = CustomImageCSVModule_kf(
        train_dir=hyperparams["TRAIN_DIR"],
        test_dir=hyperparams["TEST_DIR"],
        shape=hyperparams["SHAPE"],
        batch_size=hyperparams["BATCH_SIZE"],
        num_workers=hyperparams["NUM_WORKERS"],
        n_splits=hyperparams["K_FOLDS"],
        fold_idx=fold_idx,
    )
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for images, features, labels in test_loader:
            images, features, labels = (
                images.to(device),
                features.to(device),
                labels.to(device),
            )
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = total_loss / total_samples
    loss_list.append(avg_test_loss)

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    num_classes = len(torch.unique(all_labels))
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    precision = Precision(task="multiclass", num_classes=num_classes)
    recall = Recall(task="multiclass", num_classes=num_classes)
    f1 = F1Score(task="multiclass", num_classes=num_classes)
    conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    acc_value = accuracy(all_preds, all_labels).item()
    prec_value = precision(all_preds, all_labels).item()
    rec_value = recall(all_preds, all_labels).item()
    f1_value = f1(all_preds, all_labels).item()
    conf_matrix_value = conf_matrix(all_preds, all_labels).cpu().numpy()

    acc_list.append(acc_value)
    prec_list.append(prec_value)
    rec_list.append(rec_value)
    f1_list.append(f1_value)

    fold_metrics[fold_idx] = {
        "acc": acc_value,
        "prec": prec_value,
        "rec": rec_value,
        "f1": f1_value,
        "loss": avg_test_loss,
    }

    print(
        f"[Fold {fold_idx}] Acurácia: {acc_value:.4f} | "
        f"Precisão: {prec_value:.4f} | "
        f"Recall: {rec_value:.4f} | "
        f"Test Loss: {avg_test_loss:.4f}"
    )

    matrix_filename = model_filename.replace(".ckpt", ".png")
    matrix_path = os.path.join(model_base_dir, matrix_filename)
    plot_confusion_matrix(
        conf_matrix_value,
        save_path=matrix_path,
        title=f"Matriz de Confusão - Fold {fold_idx}",
    )

# -----------------------------------------------------------------------------#
# Estatísticas agregadas + IC 95 %
# -----------------------------------------------------------------------------#
print("\n=== Estatísticas Finais ===")
mean_acc, std_acc = print_final_stats(acc_list, "Acurácia")
mean_prec, std_prec = print_final_stats(prec_list, "Precisão")
mean_rec, std_rec = print_final_stats(rec_list, "Recall")
mean_f1, std_f1 = print_final_stats(f1_list, "F1-score")
mean_loss, std_loss = print_final_stats(loss_list, "Test Loss")

k = len(acc_list) or 1  # evita divisão por zero caso nenhum fold seja avaliado


def ci95(mean, std):
    """Retorna o intervalo de confiança (baixo, alto) para 95 % via t-Student."""
    return stats.t.interval(
        alpha=0.95, df=max(k - 1, 1), loc=mean, scale=std / np.sqrt(k)
    )


ci_acc_low, ci_acc_high = ci95(mean_acc, std_acc)
ci_prec_low, ci_prec_high = ci95(mean_prec, std_prec)
ci_rec_low, ci_rec_high = ci95(mean_rec, std_rec)
ci_f1_low, ci_f1_high = ci95(mean_f1, std_f1)
ci_loss_low, ci_loss_high = ci95(mean_loss, std_loss)

# -----------------------------------------------------------------------------#
# Gravação em arquivo .txt
# -----------------------------------------------------------------------------#
stats_filename = f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}_resultados.txt"
stats_path = os.path.join(model_base_dir, stats_filename)

with open(stats_path, "w") as f:
    # carimbo de data-hora de criação
    timestamp = datetime.now().strftime("%d/%m/%Y – %H:%M:%S")
    f.write(f"Arquivo gerado em: {timestamp}\n\n")

    # métricas por fold
    for fold, metrics in fold_metrics.items():
        f.write(f"Fold {fold}:\n")
        f.write(f"  Acurácia: {metrics['acc']:.4f}\n")
        f.write(f"  Precisão: {metrics['prec']:.4f}\n")
        f.write(f"  Recall:   {metrics['rec']:.4f}\n")
        f.write(f"  F1-score: {metrics['f1']:.4f}\n\n")
        f.write(f"  Test Loss: {metrics['loss']:.6f}\n\n")

    # médias & desvios
    f.write("=== Métricas Finais ===\n")
    f.write(f"Acurácia: Média={mean_acc:.4f}, Desvio={std_acc:.4f}\n")
    f.write(f"Precisão: Média={mean_prec:.4f}, Desvio={std_prec:.4f}\n")
    f.write(f"Recall:   Média={mean_rec:.4f}, Desvio={std_rec:.4f}\n")
    f.write(f"F1-score: Média={mean_f1:.4f}, Desvio={std_f1:.4f}\n")
    f.write(f"Test Loss: Média={mean_loss:.6f}, Desvio={std_loss:.6f}\n")

    # intervalo de confiança
    f.write("\n=== Intervalo de Confiança 95 % (t-Student) ===\n")
    f.write(f"Acurácia: [{ci_acc_low:.4f}, {ci_acc_high:.4f}]\n")
    f.write(f"Precisão: [{ci_prec_low:.4f}, {ci_prec_high:.4f}]\n")
    f.write(f"Recall:   [{ci_rec_low:.4f}, {ci_rec_high:.4f}]\n")
    f.write(f"F1-score: [{ci_f1_low:.4f}, {ci_f1_high:.4f}]\n")
    f.write(f"Test Loss: [{ci_loss_low:.6f}, {ci_loss_high:.6f}]\n")

print(f"\nResultados completos salvos em {stats_path}")
