import os
import time
import json
import torch
import pytorch_lightning as pl
import numpy as np
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# Modelo com suporte a pesos (idêntico ao original + set_class_weights)
from model_weighted import CustomEnsembleModelWeighted as CustomEnsembleModel

# Novo DataModule com StratifiedKFold + WeightedRandomSampler + class_weights
from kf_data import CustomImageCSVModule_kf_db

# (Opcional, caso você use esses callbacks em seu projeto)
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
# Seeds (agora parametrizadas)
# ---------------------------
def set_random_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _len_dataloader_safe(dl):
    """Tenta obter nº de amostras do dataloader de forma robusta."""
    try:
        return len(dl.dataset)
    except Exception:
        n = 0
        for b in dl:
            first = b[0] if isinstance(b, (list, tuple)) else b
            try:
                n += first.size(0)
            except Exception:
                n += len(first)
        return n

# ---------------------------
# Treino com K-Fold + múltiplas seeds
# ---------------------------
def train_model():
    hyperparams = load_hyperparameters('config.yaml')
    k_splits = hyperparams['K_FOLDS']
    n_seeds = hyperparams.get('N_SEEDS', 1)

    # ===== Flags opcionais para desbalanceamento (defaults seguros) =====
    use_weighted_sampler = _cfg(hyperparams, 'USE_WEIGHTED_SAMPLER', False)
    sampler_replacement  = _cfg(hyperparams, 'SAMPLER_REPLACEMENT', True)
    use_class_weights    = _cfg(hyperparams, 'USE_CLASS_WEIGHTS', False)
    weight_mode          = _cfg(hyperparams, 'WEIGHT_MODE', 'inv_freq')  # 'inv_freq' | 'effective_num'
    cb_beta              = _cfg(hyperparams, 'CB_BETA', 0.9999)          # p/ effective number (se usado)

    run_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)

    # Salva os hiperparâmetros utilizados (uma vez por execução)
    with open(os.path.join(run_dir, "hyperparams_used.json"), "w") as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)

    print("\n=== Config balanceamento (via DataModule) ===")
    print(f"USE_WEIGHTED_SAMPLER = {use_weighted_sampler}")
    print(f"SAMPLER_REPLACEMENT  = {sampler_replacement}")
    print(f"USE_CLASS_WEIGHTS    = {use_class_weights}")
    print(f"WEIGHT_MODE          = {weight_mode}")
    print(f"CB_BETA              = {cb_beta}")

    metrics_history = {}

    # ===== Loop de seeds: 42 .. 42 + N_SEEDS - 1 =====
    for seed in range(42, 42 + n_seeds):
        print(f"\n==================== Treinando com SEED {seed} ====================")
        set_random_seeds(seed)

        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # ===== Loop de folds =====
        for fold in range(k_splits):
            print(f"\n==================== Fold {fold+1}/{k_splits} ====================")

            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            fold_callback = ModelCheckpoint(
                dirpath=fold_dir,
                filename="best_model",
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

            # ===== Modelo (igual ao original + suporte a pesos) =====
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
                optimizer_momentum=(
                    (hyperparams['OPTIMIZER_MOMENTUM'], 0.999)
                    if isinstance(hyperparams['OPTIMIZER_MOMENTUM'], float)
                    else hyperparams['OPTIMIZER_MOMENTUM']
                ),
                weight_decay=hyperparams['WEIGHT_DECAY'],
                layer_scale=hyperparams['LAYER_SCALE']
            )

            # ===== DataModule (novo) com estratificação, sampler e pesos =====
            data_module = CustomImageCSVModule_kf_db(
                train_dir=hyperparams['TRAIN_DIR'],
                test_dir=hyperparams['TEST_DIR'],
                shape=hyperparams['SHAPE'],
                batch_size=hyperparams['BATCH_SIZE'],
                num_workers=hyperparams['NUM_WORKERS'],
                n_splits=k_splits,
                fold_idx=fold
            )

            # Propaga flags APENAS se o DataModule tiver os atributos (compatível)
            for attr_name, value in [
                ('use_weighted_sampler', use_weighted_sampler),
                ('sampler_replacement', sampler_replacement),
                ('use_class_weights', use_class_weights),
                ('weight_mode', weight_mode),
                ('cb_beta', cb_beta),
            ]:
                if hasattr(data_module, attr_name):
                    setattr(data_module, attr_name, value)

            # Setup normal — kf_data se encarrega do sampler e dos pesos (quando habilitado)
            data_module.setup(stage='fit')

            # Se class_weights foi calculado no DM e a flag estiver ativa, injeta no modelo
            if use_class_weights and hasattr(data_module, 'class_weights'):
                class_w = getattr(data_module, 'class_weights')
                if class_w is not None:
                    if hasattr(model, 'set_class_weights') and callable(getattr(model, 'set_class_weights')):
                        try:
                            model.set_class_weights(class_w)
                            print("Pesos de classe aplicados ao modelo via set_class_weights().")
                        except Exception as e:
                            print(f"Aviso: falha ao aplicar set_class_weights: {e}")
                    else:
                        try:
                            setattr(model, 'class_weights', class_w)
                            print("Pesos de classe anexados ao modelo (atributo class_weights).")
                        except Exception as e:
                            print(f"Aviso: não foi possível anexar class_weights ao modelo: {e}")

            callbacks = [TQDMProgressBar(leave=True), fold_callback]

            trainer = pl.Trainer(
                log_every_n_steps=10,
                accelerator=hyperparams['ACCELERATOR'],
                devices=hyperparams['DEVICES'],
                precision=hyperparams['PRECISION'],
                max_epochs=hyperparams['MAX_EPOCHS'],
                callbacks=callbacks
            )

            # ===== Tempo de treino (por fold) =====
            t0 = time.perf_counter()
            trainer.fit(model, data_module)
            train_time_sec = time.perf_counter() - t0

            best_model_path = fold_callback.best_model_path

            # Ler epoch do melhor checkpoint (se existir campo)
            checkpoint_data = torch.load(best_model_path, map_location='cpu', weights_only=False)
            best_epoch = checkpoint_data.get('epoch', None)
            print(f"Melhor modelo para seed {seed}, fold {fold} salvo na época {best_epoch}")

            # ===== Validação/Teste com o melhor checkpoint =====
            model_best = CustomEnsembleModel.load_from_checkpoint(best_model_path)

            # Garante setup de teste (se necessário)
            try:
                data_module.setup(stage='test')
            except Exception:
                pass

            # Validação
            val_metrics = trainer.validate(model_best, datamodule=data_module, verbose=False)[0]

            # Tamanho seguro do test set
            test_dl = data_module.test_dataloader()
            num_test_samples = _len_dataloader_safe(test_dl)

            # Teste + tempo + throughput
            t1 = time.perf_counter()
            test_metrics = trainer.test(model_best, datamodule=data_module, verbose=False)[0]
            test_elapsed_sec = time.perf_counter() - t1
            test_inf_ms_per_sample = (test_elapsed_sec * 1000.0 / num_test_samples) if num_test_samples > 0 else float('nan')
            test_throughput = (num_test_samples / test_elapsed_sec) if test_elapsed_sec > 0 else float('nan')

            # Memória de GPU (se disponível)
            max_gpu_mem_mb = None
            if torch.cuda.is_available():
                try:
                    max_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
                except Exception:
                    max_gpu_mem_mb = None

            # Tamanho do checkpoint
            try:
                model_size_mb = os.path.getsize(best_model_path) / 1e6
            except Exception:
                model_size_mb = None

            # Registro local por fold
            fold_txt_path = os.path.join(fold_dir, "resultados_fold.txt")
            with open(fold_txt_path, "w") as f:
                f.write(f"Seed: {seed}\n")
                f.write(f"Fold: {fold}\n")
                f.write(f"best_epoch: {best_epoch}\n")
                f.write(f"train_time_sec: {train_time_sec:.6f}\n")
                f.write(f"test_time_sec: {test_elapsed_sec:.6f}\n")
                f.write(f"test_inf_ms_per_sample: {test_inf_ms_per_sample:.6f}\n")
                f.write(f"throughput_samples_per_sec: {test_throughput:.6f}\n")
                if max_gpu_mem_mb is not None:
                    f.write(f"max_gpu_mem_mb: {max_gpu_mem_mb:.2f}\n")
                if model_size_mb is not None:
                    f.write(f"best_checkpoint_size_mb: {model_size_mb:.2f}\n")
                f.write(f"val_metrics_json: {json.dumps(val_metrics, ensure_ascii=False)}\n")
                f.write(f"test_metrics_json: {json.dumps(test_metrics, ensure_ascii=False)}\n")

            # Agrega métricas em memória (opcional)
            for metric_name, metric_value in val_metrics.items():
                metrics_history.setdefault(f"seed{seed}_val_{metric_name}", []).append(metric_value)
            for metric_name, metric_value in test_metrics.items():
                metrics_history.setdefault(f"seed{seed}_test_{metric_name}", []).append(metric_value)

            # Libera GPU
            del model, model_best
            torch.cuda.empty_cache()

    print("\n==================== Treinamento concluído ====================")

if __name__ == "__main__":
    train_model()
