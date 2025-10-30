#!/usr/bin/env python3
# benchmark_stats.py
#
# Mede custo de treino e inferência de UM fold com apenas 1 época,
# e salva estatísticas em /Documentos/projeto/treinamentos/estatisticas.
#
# O script:
# - treina por 1 época (somente para medir custo, não para ter acurácia final)
# - salva checkpoint do "melhor" modelo (val_loss mínimo)
# - mede tempo de treino, tempo de teste, latência média por imagem,
#   throughput, pico de memória GPU, tamanho do checkpoint, nº de parâmetros
# - mede métricas de teste (accuracy, f1, precision, recall, loss)
# - salva tudo em .txt e .json por fold
#
# Atenção:
# - Usa TMODEL (do YAML) para decidir qual classe de modelo construir:
#     convnext_t, swint_t               -> CustomModel
#     convnext_t_ne, swint_t_ne         -> CustomEnsembleModel
#     triple                            -> CustomModelTriple
#     vector                            -> CustomFeaturesOnlyModel
# - Usa o datamodule correspondente:
#     imagem                           -> CustomImageModule_kf
#     imagem+vetor (csv/features SSN)  -> CustomImageCSVModule_kf
#     só vetor                         -> CustomFeaturesFromFoldersModule_kf
#
# Execução sugerida:
#   python benchmark_stats.py --config config1.yaml --fold 0
#
# Você roda uma vez pra cada TMODEL (convnext_t, swint_t, ...).
#
# Saída em disco:
# /Documentos/projeto/treinamentos/estatisticas/<DATASET>_<TMODEL>/fold_<FOLD>/
#    - best_model.ckpt
#    - hyperparams_used.json
#    - benchmark_report.json
#    - benchmark_result.txt  (humano lê e cola no artigo)


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

# ===== seus modelos =====
from model import (
    CustomModel,
    CustomEnsembleModel,
    CustomModelTriple,
    CustomFeaturesOnlyModel,
)
# CustomModel          -> convnext_t / swint_t (imagem)
# CustomEnsembleModel  -> convnext_t_ne / swint_t_ne (imagem + vetor SSN)
# CustomModelTriple    -> triple (ConvNeXt + Swin + SSN)
# CustomFeaturesOnlyModel -> vector (apenas vetor SSN)


# ===== seus datamodules =====
from dataset import (
    CustomImageModule_kf,
    CustomImageCSVModule_kf,
    # CustomFeaturesFromFoldersModule_kf será importado dinamicamente
)


def load_hparams(path_yaml: str):
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
    Número total de amostras no dataloader de teste.
    Usado pra calcular latência média e throughput.
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
    Para modelos imagem+vetor, precisamos saber o tamanho do vetor de atributos
    (features_dim) pra inicializar a cabeça MLP corretamente.

    Assumimos batch = (imgs, features, labels).
    """
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, (list, tuple)) and len(batch) >= 2, \
        "Batch inesperado. Esperava (imagens, features, rótulos) para inferir features_dim."

    features = batch[1]

    # se vier lista/tupla, pega primeiro elemento como tensor
    if isinstance(features, (list, tuple)):
        features = torch.as_tensor(features[0])

    # se features for 1D, dim = tamanho do vetor
    if features.ndim == 1:
        return int(features.shape[0])

    # caso típico: (B, dim)
    return int(features.shape[-1])


def build_datamodule(h, n_splits: int, fold_idx: int):
    """
    Cria o DataModule correto com base em h["TMODEL"].

    - convnext_t, swint_t:
        usam só imagem -> CustomImageModule_kf
    - convnext_t_ne, swint_t_ne, triple:
        usam imagem + vetor SSN -> CustomImageCSVModule_kf
    - vector:
        usa só vetor SSN -> CustomFeaturesFromFoldersModule_kf
          (importado dinamicamente)
    """
    tmodel = h["TMODEL"]

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

    if tmodel in ["vector"]:
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

    raise ValueError(f"TMODEL '{tmodel}' não reconhecido para DataModule.")


def build_model(h, dm_type, dm=None):
    """
    Cria o LightningModule certo a partir de TMODEL, usando os hiperparâmetros
    do YAML. Ajusta assinatura pra cada classe do model.py.
    """
    tmodel = h["TMODEL"]

    # normaliza optimizer_momentum pra tupla
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
        # backbone puro (imagem)
        return CustomModel(
            tmodel=tmodel,
            **common_kwargs,
        )

    if tmodel in ["convnext_t_ne", "swint_t_ne"]:
        # imagem + vetor SSN
        assert dm is not None, "Preciso do datamodule pra inferir features_dim."
        features_dim = infer_features_dim_from_dm(dm)

        # repara: CustomEnsembleModel espera tmodel sem o sufixo "_ne"
        return CustomEnsembleModel(
            tmodel=tmodel.replace("_ne", ""),
            features_dim=features_dim,
            **common_kwargs,
        )

    if tmodel == "triple":
        # ConvNeXt + Swin + SSN
        assert dm is not None, "Preciso do datamodule pra inferir features_dim."
        features_dim = infer_features_dim_from_dm(dm)

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
        # só vetor SSN
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

    raise ValueError(f"TMODEL '{tmodel}' não suportado em build_model.")


def benchmark_once(h, fold_idx):
    """
    Treina 1 época só em UM fold, mede custo de treino e teste,
    e salva tudo em /Documentos/projeto/treinamentos/estatisticas/<dataset>_<tmodel>/fold_<fold>/
    """

    # ------------------------------------------------------------------
    # Diretório base fixo pedido pelo usuário
    # ------------------------------------------------------------------
    base_save_dir = "/Documentos/projeto/treinamentos/estatisticas"

    # Nome lógico "<dataset>_<tmodel>"
    run_name = f"{h['NAME_DATASET']}_{h['TMODEL']}"

    # Caminho dessa execução (modelo específico)
    run_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Pasta do fold
    fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Salvar hiperparâmetros usados (snapshot do YAML)
    with open(os.path.join(fold_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Monta datamodule pro fold
    # ------------------------------------------------------------------
    effective_n_splits = int(h.get("K_FOLDS", 5))

    dm, dm_type = build_datamodule(
        h,
        n_splits=effective_n_splits,
        fold_idx=fold_idx,
    )

    # prepara dados de treino/val
    dm.setup(stage="fit")

    # ------------------------------------------------------------------
    # Monta o modelo Lightning
    # ------------------------------------------------------------------
    model = build_model(h, dm_type, dm=dm)

    # ------------------------------------------------------------------
    # Callback pra salvar checkpoint "melhor" (pelo menor val_loss)
    # Vamos salvar exatamente dentro de fold_dir
    # ------------------------------------------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath=fold_dir,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks = [
        TQDMProgressBar(leave=True),
        ckpt_cb,
    ]

    # ------------------------------------------------------------------
    # Trainer: forçar só 1 época.
    # Ignoramos h["MAX_EPOCHS"] de propósito, porque aqui é benchmark.
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        log_every_n_steps=10,
        accelerator=h["ACCELERATOR"],
        devices=h["DEVICES"],
        precision=h["PRECISION"],
        max_epochs=1,  # <- 1 época para medir custo
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Tempo de treino + pico de memória GPU no treino
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.fit(model, dm)
    train_time_sec = time.perf_counter() - t0

    max_gpu_mb_train = None
    if torch.cuda.is_available():
        max_gpu_mb_train = torch.cuda.max_memory_allocated() / (1024**2)

    # ------------------------------------------------------------------
    # Melhor checkpoint / época salva pelo callback
    # ------------------------------------------------------------------
    best_model_path = ckpt_cb.best_model_path
    best_epoch = None
    try:
        ckpt_data = torch.load(best_model_path, map_location="cpu")
        best_epoch = ckpt_data.get("epoch", None)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Recarrega o best checkpoint pra validar/testar
    # ------------------------------------------------------------------
    model = build_model(h, dm_type, dm=dm)
    model = model.load_from_checkpoint(best_model_path)

    # Validação final (opcional, mas guardamos)
    val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]

    # ------------------------------------------------------------------
    # Teste + métricas de inferência
    # ------------------------------------------------------------------
    try:
        dm.setup(stage="test")
    except Exception:
        pass

    test_dl = dm.test_dataloader()
    n_test = len_dataloader_safe(test_dl)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t1 = time.perf_counter()
    test_metrics = trainer.test(model, datamodule=dm, verbose=False)[0]
    test_time_sec = time.perf_counter() - t1

    max_gpu_mb_test = None
    if torch.cuda.is_available():
        max_gpu_mb_test = torch.cuda.max_memory_allocated() / (1024**2)

    inf_ms_per_sample = (test_time_sec * 1000.0 / n_test) if n_test > 0 else float("nan")
    throughput_samples_per_sec = (n_test / test_time_sec) if test_time_sec > 0 else float("nan")

    # ------------------------------------------------------------------
    # Tamanho do checkpoint e nº de parâmetros
    # ------------------------------------------------------------------
    checkpoint_size_mb = None
    if os.path.isfile(best_model_path):
        checkpoint_size_mb = os.path.getsize(best_model_path) / (1024**2)

    num_params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # ------------------------------------------------------------------
    # Monta dicionário final de estatísticas
    # ------------------------------------------------------------------
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
        "num_params_trainable": num_params_trainable,

        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    # ------------------------------------------------------------------
    # Salvar benchmark_report.json (máquina lê)
    # ------------------------------------------------------------------
    json_path = os.path.join(fold_dir, "benchmark_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Salvar benchmark_result.txt (humano lê / cola no artigo)
    # ------------------------------------------------------------------
    txt_path = os.path.join(fold_dir, "benchmark_result.txt")
    with open(txt_path, "w") as f:
        f.write("===== BENCHMARK REPORT =====\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    # print no console também, só pra feedback imediato
    print("\n===== BENCHMARK REPORT =====")
    print(f"(salvo em {txt_path})")
    for k, v in report.items():
        print(f"{k}: {v}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config1.yaml",
        help="Arquivo YAML com hiperparâmetros (ex.: config1.yaml)."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold que será usado para o benchmark (0..K_FOLDS-1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed (para reprodutibilidade)."
    )

    args = parser.parse_args()

    # carrega hiperparâmetros
    h = load_hparams(args.config)

    # força semente
    set_seeds(args.seed)

    # roda benchmark no fold escolhido
    benchmark_once(h, fold_idx=args.fold)
