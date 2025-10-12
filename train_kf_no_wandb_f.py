import os
import time
import json
import torch
import pytorch_lightning as pl
import numpy as np
import yaml
import random
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from model import CustomEnsembleModel
from kf_data import CustomImageCSVModule_kf
from callbacks import (
    EarlyStoppingAtSpecificEpoch,
    SaveBestOrLastModelCallback,
    EarlyStopCallback
)

# Carregar hiperparâmetros do arquivo config.yaml
def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparams = yaml.safe_load(file)
    return hyperparams

# Configurar sementes para garantir reprodutibilidade
def set_random_seeds(seed):
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
        # fallback: soma tamanhos de batch em uma passada rápida
        n = 0
        for b in dl:
            # b pode ser (x,y) ou (x,features,y). Contar pelo 1º tensor
            if isinstance(b, (list, tuple)):
                first = b[0]
            else:
                first = b
            try:
                n += first.size(0)
            except Exception:
                # se não for tensor, tenta len()
                n += len(first)
        return n

# Função principal para treinamento com validação cruzada
def train_model():
    hyperparams = load_hyperparameters('config.yaml')
    k_splits = hyperparams['K_FOLDS']
    n_seeds = hyperparams.get('N_SEEDS', 1)  # padrão = 1 se não definido

    # Diretório raiz do experimento (inalterado, só assegurado)
    run_dir = os.path.join("modelos_kf", f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)

    # Salva uma cópia dos hiperparâmetros usados (reprodutibilidade)
    with open(os.path.join(run_dir, "hyperparams_used.json"), "w") as f:
        json.dump(hyperparams, f, indent=2, ensure_ascii=False)

    # Arquivo único de resultados por experimento
    resultados_path = os.path.join(
        run_dir,
        f"{hyperparams['NAME_DATASET']}_{hyperparams['TMODEL']}_resultados.txt"
    )
    # Cabeçalho do arquivo (idempotente)
    if not os.path.exists(resultados_path):
        with open(resultados_path, "w") as f:
            f.write("# seed\tfold\tbest_epoch\ttrain_time_sec\ttest_inf_ms_per_sample\t" +
                    "VAL_METRICS(json)\tTEST_METRICS(json)\n")

    # Dicionário para acumular métricas de todas as (seed, fold)
    metrics_history = {}

    for seed in range(42, 42 + n_seeds):
        print(f"\n==================== Treinando com SEED {seed} ====================")
        set_random_seeds(seed)

        # diretório da seed
        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for fold in range(k_splits):
            print(f"\n==================== Fold {fold+1}/{k_splits} ====================")

            # diretório seed/fold
            fold_dir = os.path.join(seed_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            fold_callback = ModelCheckpoint(
                dirpath=fold_dir,                 # salva em seed/fold
                filename="best_model",            # best_model.ckpt
                monitor="val_loss",
                mode="min",
                save_top_k=1
            )

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

            data_module = CustomImageCSVModule_kf(
                train_dir=hyperparams['TRAIN_DIR'],
                test_dir=hyperparams['TEST_DIR'],
                shape=hyperparams['SHAPE'],
                batch_size=hyperparams['BATCH_SIZE'],
                num_workers=hyperparams['NUM_WORKERS'],
                n_splits=k_splits,
                fold_idx=fold
            )
            data_module.setup(stage='fit')

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

            # -----------------------
            # Tempo de treino (fold)
            # -----------------------
            t0 = time.perf_counter()
            trainer.fit(model, data_module)
            train_time_sec = time.perf_counter() - t0

            best_model_path = fold_callback.best_model_path

            # Carregar epoch do melhor modelo (deixa explícito p/ futuro default do torch)
            checkpoint_data = torch.load(best_model_path, map_location='cpu', weights_only=False)  # >>>
            best_epoch = checkpoint_data.get('epoch', None)
            print(f"Melhor modelo para seed {seed}, fold {fold} salvo na época {best_epoch}")

            # Avaliação (val e test) no melhor checkpoint
            model = CustomEnsembleModel.load_from_checkpoint(best_model_path)

            # >>> preparar split de teste (evita AttributeError: test_ds)
            try:
                data_module.setup(stage='test')
            except Exception:
                pass

            # Validation
            val_metrics = trainer.validate(model, datamodule=data_module, verbose=False)[0]

            # Test + tempo médio de inferência
            # Obtém nº de amostras de teste
            test_dl = data_module.test_dataloader()
            num_test_samples = _len_dataloader_safe(test_dl)

            t1 = time.perf_counter()
            test_metrics = trainer.test(model, datamodule=data_module, verbose=False)[0]
            test_elapsed_sec = time.perf_counter() - t1

            # ms por amostra (estimativa geral do loop de teste)
            test_inf_ms_per_sample = (test_elapsed_sec * 1000.0 / num_test_samples) if num_test_samples > 0 else float('nan')  # >>>
            # throughput (amostras/s)
            test_throughput = (num_test_samples / test_elapsed_sec) if test_elapsed_sec > 0 else float('nan')  # >>>
            # pico de memória (MB) e tamanho do ckpt (MB) — best effort
            max_gpu_mem_mb = None  # >>>
            if torch.cuda.is_available():
                try:
                    max_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
                except Exception:
                    max_gpu_mem_mb = None
            try:
                model_size_mb = os.path.getsize(best_model_path) / 1e6  # >>>
            except Exception:
                model_size_mb = None

            # Acumula métricas para estatística ao final (mantido)
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float, np.floating)):
                    metrics_history.setdefault(f"val/{metric_name}", []).append(float(metric_value))
            for metric_name, metric_value in test_metrics.items():
                if isinstance(metric_value, (int, float, np.floating)):
                    metrics_history.setdefault(f"test/{metric_name}", []).append(float(metric_value))

            metrics_history.setdefault("train_time_sec", []).append(float(train_time_sec))
            metrics_history.setdefault("test_inf_ms_per_sample", []).append(float(test_inf_ms_per_sample))

            # Registro por linha no arquivo consolidado (mantido)
            with open(resultados_path, "a") as f:
                f.write(
                    f"{seed}\t{fold}\t{best_epoch}\t"
                    f"{train_time_sec:.6f}\t{test_inf_ms_per_sample:.6f}\t"
                    f"{json.dumps(val_metrics, ensure_ascii=False)}\t"
                    f"{json.dumps(test_metrics, ensure_ascii=False)}\n"
                )

            # >>> Novo: salvar um .txt por FOLD (sobrescreve a cada execução)
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

            # Libera GPU
            del model
            torch.cuda.empty_cache()

    # ====================
    # Resumo final: mean±std
    # ====================
    print("\n==================== Métricas Finais ====================")
    with open(resultados_path, "a") as f:
        f.write("\n# Resumo (mean ± std)\n")
        for metric_name, values in metrics_history.items():
            if len(values) == 0:
                continue
            if isinstance(values[0], (int, float, np.floating)):
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=0))
                print(f"{metric_name}: mean = {mean:.4f}, std = {std:.4f}")
                f.write(f"# {metric_name}\tmean={mean:.6f}\tstd={std:.6f}\n")

if __name__ == "__main__":
    train_model()
