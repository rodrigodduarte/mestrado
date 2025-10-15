import os
import shutil
import torch
import pytorch_lightning as pl
import numpy as np
import yaml
import random
from typing import Sequence, List, Optional

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback
)

# ---------------------------
# Utilidades de configuração
# ---------------------------
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

def _cfg(hparams: dict, key: str, default):
    """Lê chave do config com valor padrão (compatível se a chave não existir)."""
    return hparams[key] if key in hparams else default

# ---------------------------
# Seeds (mantido)
# ---------------------------
def set_random_seeds():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ---------------------------
# Helpers para desbalanceamento
# ---------------------------
def _unwrap_dataset(ds):
    """Desempacota Subset(s) aninhados até chegar no dataset base, retornando (base_ds, indices)"""
    indices = None
    while isinstance(ds, Subset):
        indices = ds.indices if indices is None else [ds.indices[i] for i in indices]
        ds = ds.dataset
    return ds, indices

def _extract_labels(dataset) -> List[int]:
    """
    Extrai rótulos do dataset de forma robusta:
    - Tenta atributos comuns: .targets, .labels, .y
    - Se for Subset, usa os índices
    - Fallback (evitar): itera __getitem__ uma vez por índice (apenas se necessário)
    """
    base, subset_idx = _unwrap_dataset(dataset)

    # Alvos por atributo comum
    for attr in ("targets", "labels", "y"):
        if hasattr(base, attr):
            base_labels = list(getattr(base, attr))
            if subset_idx is not None:
                return [base_labels[i] for i in subset_idx]
            return base_labels

    # Alguns datasets (ImageFolder) guardam em .samples (path, class_idx)
    if hasattr(base, "samples"):
        base_labels = [c for _, c in base.samples]
        if subset_idx is not None:
            return [base_labels[i] for i in subset_idx]
        return base_labels

    # Fallback (mais custoso): consultar __getitem__
    if subset_idx is None:
        subset_idx = list(range(len(base)))
    labels = []
    for i in subset_idx:
        item = base[i]
        # Suporta (x, y) ou (x, features, y)
        if isinstance(item, (list, tuple)):
            y = item[-1]
        else:
            raise RuntimeError("Não foi possível extrair rótulos do dataset.")
        labels.append(int(y))
    return labels

def _class_weights_from_counts(counts: np.ndarray, mode: str = "inv_freq", beta: float = 0.9999) -> np.ndarray:
    """
    Gera pesos por classe a partir das contagens:
    - inv_freq: peso_c = 1 / (count_c + eps)
    - effective_num (Cui et al.): w_c = (1 - beta) / (1 - beta^{count_c})
    """
    eps = 1e-12
    counts = counts.astype(np.float64)
    if mode == "effective_num":
        # evita divisão por zero
        w = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0)))
    else:
        w = 1.0 / (counts + eps)
    # normaliza para média = 1 (estável para otimizador)
    w = w * (len(w) / (w.sum() + eps))
    return w

def _make_weighted_train_loader(original_loader: DataLoader,
                                labels: Sequence[int],
                                replacement: bool = True) -> DataLoader:
    """Cria um DataLoader novo para treino usando WeightedRandomSampler, mantendo parâmetros do loader original."""
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes)
    # pesos por amostra: inversamente proporcionais à frequência da classe
    inv_freq = 1.0 / np.maximum(counts, 1)
    sample_weights = inv_freq[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=replacement
    )

    # Copia parâmetros úteis do loader original (quando disponíveis)
    kwargs = dict(
        dataset=original_loader.dataset,
        batch_size=original_loader.batch_size,
        sampler=sampler,
        shuffle=False,  # sampler e shuffle não devem coexistir
        num_workers=original_loader.num_workers,
        pin_memory=getattr(original_loader, "pin_memory", False),
        drop_last=getattr(original_loader, "drop_last", False),
        persistent_workers=getattr(original_loader, "persistent_workers", False)
    )
    if hasattr(original_loader, "collate_fn") and original_loader.collate_fn is not None:
        kwargs["collate_fn"] = original_loader.collate_fn
    if hasattr(original_loader, "prefetch_factor") and original_loader.prefetch_factor is not None:
        kwargs["prefetch_factor"] = original_loader.prefetch_factor

    return DataLoader(**kwargs)

# ---------------------------
# Treino com K-Fold
# ---------------------------
def train_model():
    hyperparams = load_hyperparameters('config.yaml')
    k_splits = hyperparams['K_FOLDS']
    metrics_history = {}

    # ===== Flags opcionais para desbalanceamento =====
    use_weighted_sampler = _cfg(hyperparams, 'USE_WEIGHTED_SAMPLER', False)
    sampler_replacement = _cfg(hyperparams, 'SAMPLER_REPLACEMENT', True)
    use_class_weights   = _cfg(hyperparams, 'USE_CLASS_WEIGHTS', False)
    weight_mode         = _cfg(hyperparams, 'WEIGHT_MODE', 'inv_freq')  # 'inv_freq' | 'effective_num'
    cb_beta             = _cfg(hyperparams, 'CB_BETA', 0.9999)          # p/ effective number, se usado

    run_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)

    print("\n=== Config balanceamento (compatível) ===")
    print(f"USE_WEIGHTED_SAMPLER = {use_weighted_sampler}")
    print(f"SAMPLER_REPLACEMENT  = {sampler_replacement}")
    print(f"USE_CLASS_WEIGHTS    = {use_class_weights}")
    print(f"WEIGHT_MODE          = {weight_mode}")
    print(f"CB_BETA              = {cb_beta}")

    for fold in range(k_splits):
        print(f"\n==================== Fold {fold+1}/{k_splits} ====================")

        fold_callback = ModelCheckpoint(
            dirpath=run_dir,
            filename=f"fold_{fold}_best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        # Modelo (inalterado) — compatível com futura injeção de pesos
        model = CustomEnsembleModel(
            tmodel=hyperparams["TMODEL"],
            name_dataset=hyperparams["NAME_DATASET"],
            shape=hyperparams["SHAPE"],
            epochs=hyperparams['MAX_EPOCHS'],
            learning_rate=hyperparams['LEARNING_RATE'],
            features_dim=hyperparams["FEATURES_DIM"],
            drop_path_rate=hyperparams['DROP_PATH_RATE'],
            num_classes=hyperparams['NUM_CLASSES'],
            label_smoothing=hyperparams['LABEL_SMOOTHING'],
            optimizer_momentum=(hyperparams['OPTIMIZER_MOMENTUM'], 0.999),
            weight_decay=hyperparams['WEIGHT_DECAY'],
            layer_scale=hyperparams['LAYER_SCALE']
        )

        # DataModule (inalterado)
        data_module = CustomImageCSVModule_kf(
            train_dir=hyperparams['TRAIN_DIR'],
            test_dir=hyperparams['TEST_DIR'],
            shape=hyperparams['SHAPE'],
            batch_size=hyperparams['BATCH_SIZE'],
            num_workers=hyperparams['NUM_WORKERS'],
            n_splits=k_splits,
            fold_idx=fold
        )

        # 1) Setup padrão do DM para obter loaders originais
        data_module.setup(stage='fit')

        # 2) Substituir APENAS o train_dataloader por um com WeightedRandomSampler (se habilitado)
        if use_weighted_sampler:
            try:
                original_train_loader = data_module.train_dataloader()
                train_labels = _extract_labels(original_train_loader.dataset)
                # Recria loader com sampler ponderado
                weighted_train_loader = _make_weighted_train_loader(
                    original_loader=original_train_loader,
                    labels=train_labels,
                    replacement=sampler_replacement
                )

                # Monkey patch: mantém a API Trainer.fit(model, datamodule),
                # mas devolve nosso loader ponderado quando o Trainer chamar train_dataloader()
                def _patched_train_loader():
                    return weighted_train_loader
                setattr(data_module, "train_dataloader", _patched_train_loader)

                print(f"[imbalance] WeightedRandomSampler ativado no fold {fold}.")
            except Exception as e:
                print(f"[imbalance][aviso] Falha ao ativar WeightedRandomSampler no fold {fold}: {e}")

        # 3) Injetar class_weights na loss do modelo (se habilitado)
        if use_class_weights:
            try:
                # Sempre extraímos os labels do treino para calcular os pesos de classe
                # (mesmo que o sampler esteja desabilitado)
                tl_for_weights = data_module.train_dataloader()
                labels_for_weights = _extract_labels(tl_for_weights.dataset)
                labels_for_weights = np.asarray(labels_for_weights)
                num_classes = int(labels_for_weights.max()) + 1
                counts = np.bincount(labels_for_weights, minlength=num_classes)
                cls_w = _class_weights_from_counts(counts, mode=weight_mode, beta=cb_beta)
                cls_w_t = torch.tensor(cls_w, dtype=torch.float32)

                if hasattr(model, 'set_class_weights') and callable(getattr(model, 'set_class_weights')):
                    model.set_class_weights(cls_w_t)
                    print(f"[imbalance] class_weights aplicados via set_class_weights() no fold {fold}.")
                else:
                    setattr(model, 'class_weights', cls_w_t)
                    print(f"[imbalance] class_weights anexados ao modelo (atributo) no fold {fold}.")
            except Exception as e:
                print(f"[imbalance][aviso] Falha ao calcular/aplicar class_weights no fold {fold}: {e}")

        callbacks = [
            TQDMProgressBar(leave=True),
            fold_callback
        ]

        trainer = pl.Trainer(
            log_every_n_steps=10,
            accelerator=hyperparams['ACCELERATOR'],
            devices=hyperparams['DEVICES'],
            precision=hyperparams['PRECISION'],
            max_epochs=hyperparams['MAX_EPOCHS'],
            callbacks=callbacks
        )

        # Mantido: usamos o DataModule (agora com train_dataloader possivelmente "patchado")
        trainer.fit(model, data_module)

        best_model_path = fold_callback.best_model_path

        # Mantido: epoch do melhor modelo
        checkpoint_data = torch.load(best_model_path, map_location='cpu')
        best_epoch = checkpoint_data['epoch']
        print(f"Melhor modelo para fold {fold} salvo na época {best_epoch}")

        model = CustomEnsembleModel.load_from_checkpoint(best_model_path)
        val_metrics = trainer.validate(model, data_module)[0]
        test_metrics = trainer.test(model, data_module)[0]

        for metric_name, metric_value in val_metrics.items():
            if metric_name not in metrics_history:
                metrics_history[metric_name] = []
            metrics_history[metric_name].append(metric_value)

        for metric_name, metric_value in test_metrics.items():
            if metric_name not in metrics_history:
                metrics_history[metric_name] = []
            metrics_history[metric_name].append(metric_value)

        # Limpar modelo da GPU ao final do fold
        del model
        torch.cuda.empty_cache()

    print("\n==================== Métricas Finais ====================")
    for metric_name, values in metrics_history.items():
        if isinstance(values[0], (int, float, np.float32, np.float64)):
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric_name}: mean = {mean:.4f}, std = {std:.4f}")

if __name__ == "__main__":
    set_random_seeds()
    train_model()
