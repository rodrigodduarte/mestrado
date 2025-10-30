#!/usr/bin/env python3
# benchmark_stats.py
#
# Mede custo de treino e inferência de UM fold rapidamente (ex.: 1 época),
# e salva estatísticas úteis para o artigo:
# - tempo de treino
# - tempo de teste
# - latência média por imagem
# - throughput (img/s)
# - pico de memória GPU
# - tamanho do checkpoint
# - #parâmetros
# - métricas de teste (accuracy, f1, precision, recall, loss)
#
# Baseado nos padrões dos seus scripts de treino k-fold em Lightning
# e nos módulos/modelos já existentes. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
# e também em train_ensemble_csv_kf.py / train_images_kf_only.py. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
#
# Uso típico:
#   python benchmark_stats.py --config config1.yaml --fold 0 --out resultados_benchmark.txt
#
# Ajuste TMODEL no YAML para trocar entre:
#   - "convnext_t", "swint_t"                  -> CustomModel (imagem)
#   - "convnext_t_ne", "swint_t_ne"            -> CustomEnsembleModel (imagem + features SSN)
#   - "triple"                                 -> CustomModelTriple (ConvNeXt+Swin+SSN)
#   - "vector"                                 -> CustomFeaturesOnlyModel (só vetor)
#
# IMPORTANTE:
# - Este script roda só 1 fold e apenas 1 época (benchmark).
# - Ele cria uma pasta ./benchmark_tmp/<dataset>_<tmodel>/fold_<fold>/
#   para salvar o checkpoint best_model.ckpt.
#
# Você pode rodar vários modelos (cada um com seu TMODEL) e depois comparar
# os .txt gerados para escrever a parte de custo computacional no artigo.


import os
import time
import json
import yaml
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# ===== importa seus modelos =====
from model import (
    CustomModel,
    CustomEnsembleModel,
    CustomModelTriple,
    CustomFeaturesOnlyModel,
)
# Esses nomes vêm do seu arquivo model.py, que define as quatro variantes:
# - CustomModel          (ConvNeXt_t ou Swin_t puros)
# - CustomEnsembleModel  (ConvNeXt/Swin + vetor SSN)
# - CustomModelTriple    (ConvNeXt + Swin + vetor SSN)
# - CustomFeaturesOnlyModel (apenas vetor SSN) :contentReference[oaicite:10]{index=10}


# ===== importa seus datamodules =====
from dataset import (
    CustomImageModule_kf,
    CustomImageCSVModule_kf,
    # CustomFeaturesFromFoldersModule_kf  <- definido no seu código de features-only
    #                                     (aparece no train_kf_no_wandb_f.py) :contentReference[oaicite:11]{index=11}
    # Se estiver em outro arquivo, importe daqui ou ajuste o nome.
)

###############################################################################
# Helpers utilitários (inspirados nos seus scripts) :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}
###############################################################################

def load_hparams(path_yaml):
    with open(path_yaml, "r") as f:
        return yaml.safe_load(f)

def set_seeds(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def len_dataloader_safe(dl):
    """
    Conta o número total de amostras no dataloader de teste.
    Isso é usado pra calcular latência média e throughput.
    """
    try:
        return len(dl.dataset)
    except Exception:
        n = 0
        for b in dl:
            if isinstance(b, (list, tuple)) and len(b) >= 1:
                first = b[0]
            else:
                first = b
            try:
                n += first.size(0)
            except Exception:
                try:
                    n += len(first)
                except Exception:
                    n += 1
        return n

def infer_features_dim_from_dm(dm):
    """
    Para modelos que usam imagem + vetor de características (ex.: CustomEnsembleModel,
    CustomModelTriple), precisamos saber features_dim pra inicializar a cabeça.
    Isso segue a lógica de _infer_features_dim() do seu train_ensemble_csv_kf.py. :contentReference[oaicite:14]{index=14}
    """
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # batch esperado: (images, features, labels)
    assert isinstance(batch, (list, tuple)) and len(batch) >= 2, \
        "Esperava batch tipo (img, features, label) para inferir features_dim"
    features = batch[1]
    if isinstance(features, (list, tuple)):
        features = torch.as_tensor(features[0])
    if features.ndim == 1:
        return int(features.shape[0])
    return int(features.shape[-1])

###############################################################################
# Construção dinâmica do DataModule de acordo com TMODEL
###############################################################################

def build_datamodule(h, n_splits: int, fold_idx: int):
    """
    Escolhe o DataModule certo a partir de TMODEL.
    - Modelos puros de imagem (convnext_t, swint_t) usam CustomImageModule_kf.
    - Modelos que combinam imagem+vetor (convnext_t_ne, swint_t_ne, triple)
      usam CustomImageCSVModule_kf (imagem + CSV de features).
    - Modelo só-vetor ("vector") usa CustomFeaturesFromFoldersModule_kf,
      que está no seu pipeline de SSN puro. :contentReference[oaicite:15]{index=15}
    """
    tmodel = h["TMODEL"]

    # Caso 1: apenas imagem
    if tmodel in ["convnext_t", "swint_t"]:
        dm = CustomImageModule_kf(
            train_dir=h["TRAIN_DIR"],
            test_dir=h["TEST_DIR"],
            shape=h["SHAPE"],
            batch_size=h["BATCH_SIZE"],
            num_workers=h["NUM_WORKERS"],
            n_splits=n_splits,
            fold_idx=fold_idx,
        )
        return dm, "image_only"

    # Caso 2: imagem + vetor (concatenação)
    if tmodel in ["convnext_t_ne", "swint_t_ne", "triple"]:
        dm = CustomImageCSVModule_kf(
            train_dir=h["TRAIN_DIR"],
            test_dir=h["TEST_DIR"],
            shape=h["SHAPE"],
            batch_size=h["BATCH_SIZE"],
            num_workers=h["NUM_WORKERS"],
            n_splits=n_splits,
            fold_idx=fold_idx,
        )
        return dm, "image_plus_features"

    # Caso 3: apenas vetor de atributos SSN
    if tmodel in ["vector"]:
        # Esta classe é usada no seu script train_kf_no_wandb_f.py. :contentReference[oaicite:16]{index=16}
        # Importamos aqui dentro pra evitar erro se ela não existir em dataset.py atual.
        from dataset import CustomFeaturesFromFoldersModule_kf
        dm = CustomFeaturesFromFoldersModule_kf(
            train_dir=h["TRAIN_DIR"],
            test_dir=h["TEST_DIR"],
            shape=h["SHAPE"],
            batch_size=h["BATCH_SIZE"],
            num_workers=h["NUM_WORKERS"],
            n_splits=n_splits,
            fold_idx=fold_idx,
        )
        return dm, "features_only"

    raise ValueError(f"TMODEL '{tmodel}' não reconhecido para construção do DataModule.")


###############################################################################
# Construção dinâmica do modelo de acordo com TMODEL
###############################################################################

def build_model(h, dm_type, dm=None):
    """
    Cria a LightningModule certa com base em TMODEL.
    - convnext_t, swint_t          -> CustomModel (usa só imagem)
    - convnext_t_ne, swint_t_ne    -> CustomEnsembleModel (imagem + vetor SSN)
    - triple                       -> CustomModelTriple (ConvNeXt + Swin + vetor SSN)
    - vector                       -> CustomFeaturesOnlyModel (só vetor SSN)
    Usa os hiperparâmetros do YAML (config1.yaml). :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}
    """
    tmodel = h["TMODEL"]

    # normalizar OPTIMIZER_MOMENTUM em tupla (vezes você salva como float único)
    opt_mom = h["OPTIMIZER_MOMENTUM"]
    if isinstance(opt_mom, (int, float)):
        opt_mom = (opt_mom, 0.999)
    elif isinstance(opt_mom, list):
        opt_mom = tuple(opt_mom)

    common_kwargs = dict(
        name_dataset=h["NAME_DATASET"],
        shape=h["SHAPE"],
        epochs=h["MAX_EPOCHS"],
        learning_rate=h["LEARNING_RATE"],
        drop_path_rate=h["DROP_PATH_RATE"],
        num_classes=h["NUM_CLASSES"],
        label_smoothing=h["LABEL_SMOOTHING"],
        optimizer_momentum=opt_mom,
        weight_decay=h["WEIGHT_DECAY"],
        layer_scale=h["LAYER_SCALE"],
    )

    if tmodel in ["convnext_t", "swint_t"]:
        # backbone puro: usa CustomModel (imagem-only). :contentReference[oaicite:19]{index=19}
        return CustomModel(
            tmodel=tmodel,
            **common_kwargs,
        )

    if tmodel in ["convnext_t_ne", "swint_t_ne"]:
        # ensemble imagem+vetor: CustomEnsembleModel precisa de features_dim detectado. :contentReference[oaicite:20]{index=20}
        assert dm is not None, "Preciso do dm para inferir features_dim"
        features_dim = infer_features_dim_from_dm(dm)
        return CustomEnsembleModel(
            tmodel=tmodel.replace("_ne", ""),  # backbone base: convnext_t ou swint_t
            features_dim=features_dim,
            **common_kwargs,
        )

    if tmodel == "triple":
        # triple usa CustomModelTriple, que combina ConvNeXt + Swin + vetor. :contentReference[oaicite:21]{index=21}
        assert dm is not None, "Preciso do dm para inferir features_dim"
        features_dim = infer_features_dim_from_dm(dm)
        # CustomModelTriple tem assinatura levemente diferente
        return CustomModelTriple(
            name_dataset=h["NAME_DATASET"],
            shape=h["SHAPE"],
            epochs=h["MAX_EPOCHS"],
            learning_rate=h["LEARNING_RATE"],
            features_dim=features_dim,
            drop_path_rate=h["DROP_PATH_RATE"],
            num_classes=h["NUM_CLASSES"],
            label_smoothing=h["LABEL_SMOOTHING"],
            optimizer_momentum=opt_mom,
            weight_decay=h["WEIGHT_DECAY"],
            layer_scale=h["LAYER_SCALE"],
        )

    if tmodel == "vector":
        # modelo só de vetor SSN. :contentReference[oaicite:22]{index=22}
        return CustomFeaturesOnlyModel(
            name_dataset=h["NAME_DATASET"],
            shape=h["SHAPE"],
            epochs=h["MAX_EPOCHS"],
            learning_rate=h["LEARNING_RATE"],
            features_dim=h["FEATURES_DIM"],
            drop_path_rate=h["DROP_PATH_RATE"],
            num_classes=h["NUM_CLASSES"],
            label_smoothing=h["LABEL_SMOOTHING"],
            optimizer_momentum=opt_mom,
            weight_decay=h["WEIGHT_DECAY"],
            layer_scale=h["LAYER_SCALE"],
        )

    raise ValueError(f"TMODEL '{tmodel}' não suportado.")


###############################################################################
# Benchmark principal
###############################################################################

def benchmark_once(h, fold_idx, out_txt):
    """
    Roda UM fold com UMA época, mede custo de treino/inferência e salva estatísticas.
    """

    # --------------------------------------------------
    # Setup pasta de saída
    # --------------------------------------------------
    run_dir = os.path.join("benchmark_tmp", f"{h['NAME_DATASET']}_{h['TMODEL']}")
    os.makedirs(run_dir, exist_ok=True)
    fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Salva hiperparâmetros usados
    with open(os.path.join(fold_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # Instanciar DataModule
    # Aqui usamos sempre K_FOLDS=5, e escolhemos o fold_idx pedido.
    # Isso é equivalente a "pegar um fold representativo".
    # --------------------------------------------------
    effective_n_splits = int(h.get("K_FOLDS", 5))
    dm, dm_type = build_datamodule(h, n_splits=effective_n_splits, fold_idx=fold_idx)

    # preparar dados de treino/val
    dm.setup(stage="fit")

    # --------------------------------------------------
    # Instanciar modelo
    # --------------------------------------------------
    model = build_model(h, dm_type, dm=dm)

    # --------------------------------------------------
    # Callbacks para salvar melhor checkpoint
    # Vamos monitorar 'val_loss' e salvar sempre o melhor dentro de fold_dir
    # --------------------------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath=fold_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [TQDMProgressBar(leave=True), ckpt_cb]

    # --------------------------------------------------
    # Trainer com 1 época só (benchmark!)
    # IMPORTANTE: aqui eu ignoro h["MAX_EPOCHS"] e forço max_epochs=1.
    # --------------------------------------------------
    trainer = pl.Trainer(
        log_every_n_steps=10,
        accelerator=h["ACCELERATOR"],
        devices=h["DEVICES"],
        precision=h["PRECISION"],
        max_epochs=1,  # <--- só 1 época para benchmark
        callbacks=callbacks,
    )

    # --------------------------------------------------
    # Medir tempo de treino
    # (Isso inclui forward/backward/otimizador dessa 1 época.)
    # --------------------------------------------------
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    trainer.fit(model, dm)
    train_time_sec = time.perf_counter() - t0

    # pico de memória durante treino (MB)
    max_gpu_mb_train = None
    if torch.cuda.is_available():
        max_gpu_mb_train = torch.cuda.max_memory_allocated() / (1024**2)

    # --------------------------------------------------
    # Melhor checkpoint e época
    # --------------------------------------------------
    best_model_path = ckpt_cb.best_model_path
    best_epoch = None
    try:
        ckpt_data = torch.load(best_model_path, map_location="cpu")
        best_epoch = ckpt_data.get("epoch", None)
    except Exception:
        pass

    # --------------------------------------------------
    # Recarregar modelo salvo do best checkpoint
    # para medir validação e teste
    # --------------------------------------------------
    # Precisamos reconstruir com os mesmos kwargs.
    model = build_model(h, dm_type, dm=dm)
    model = model.load_from_checkpoint(best_model_path)

    # --------------------------------------------------
    # Validação final (só pra logar)
    # --------------------------------------------------
    val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]

    # --------------------------------------------------
    # Teste + tempos de inferência
    # --------------------------------------------------
    # garantir que o datamodule saiba montar o test_dl
    try:
        dm.setup(stage="test")
    except Exception:
        pass

    test_dl = dm.test_dataloader()
    n_test = len_dataloader_safe(test_dl)

    # zera estatística de pico de memória antes do teste
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t1 = time.perf_counter()
    test_metrics = trainer.test(model, datamodule=dm, verbose=False)[0]
    test_time_sec = time.perf_counter() - t1

    # pico de memória durante teste (MB)
    max_gpu_mb_test = None
    if torch.cuda.is_available():
        max_gpu_mb_test = torch.cuda.max_memory_allocated() / (1024**2)

    inf_ms_per_sample = (test_time_sec * 1000.0 / n_test) if n_test > 0 else float("nan")
    throughput_samples_per_sec = (n_test / test_time_sec) if test_time_sec > 0 else float("nan")

    # --------------------------------------------------
    # Estatísticas do arquivo salvo
    # --------------------------------------------------
    checkpoint_size_mb = None
    if os.path.isfile(best_model_path):
        checkpoint_size_mb = os.path.getsize(best_model_path) / (1024**2)

    # número de parâmetros treináveis
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --------------------------------------------------
    # Montar relatório
    # OBS: test_metrics deve conter chaves como
    # "test_accuracy", "test_f1", "test_precision", "test_recall", "test_loss"
    # pois seus LightningModules logam isso no test_step. :contentReference[oaicite:23]{index=23}
    # --------------------------------------------------
    report = {
        "dataset": h["NAME_DATASET"],
        "tmodel": h["TMODEL"],
        "fold_idx": fold_idx,

        "best_epoch": best_epoch,
        "train_time_sec": train_time_sec,
        "test_time_sec": test_time_sec,

        "test_inf_ms_per_sample": inf_ms_per_sample,
        "throughput_samples_per_sec": throughput_samples_per_sec,

        "max_gpu_mem_mb_train": max_gpu_mb_train,
        "max_gpu_mem_mb_test": max_gpu_mb_test,

        "checkpoint_size_mb": checkpoint_size_mb,
        "num_params_trainable": n_params,

        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # imprime no console
    print("\n===== BENCHMARK REPORT =====")
    for k, v in report.items():
        print(f"{k}: {v}")

    # salva em txt pra você usar no artigo
    with open(out_txt, "w") as f:
        f.write("===== BENCHMARK REPORT =====\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    # também salva JSON bruto na pasta do fold pra rastreabilidade
    with open(os.path.join(fold_dir, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config1.yaml",
                        help="YAML de hiperparâmetros (ex.: config1.yaml).")
    parser.add_argument("--fold", type=int, default=0,
                        help="Qual fold medir (0..K_FOLDS-1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed usada pra reprodutibilidade.")
    parser.add_argument("--out", type=str, default="benchmark_result.txt",
                        help="Arquivo .txt de saída com o resumo.")
    args = parser.parse_args()

    h = load_hparams(args.config)

    # força 1 seed pra benchmark
    set_seeds(args.seed)

    benchmark_once(h, fold_idx=args.fold, out_txt=args.out)
