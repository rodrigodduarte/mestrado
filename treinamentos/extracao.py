#!/usr/bin/env python3
# extracao_all.py
#
# Benchmark automático:
# - Usa SOMENTE treinamentos/config1.yaml
# - Para cada modelo profundo (ConvNeXt, Swin, ConvNeXt+SSN, Swin+SSN, Triple)
# - Para cada fold (0..K_FOLDS-1)
#   - treina 1 época só (suficiente pra medir custo computacional)
#   - mede tempo de treino, tempo de teste, latência média, throughput
#   - mede pico de memória GPU treino/teste
#   - mede tamanho do checkpoint e nº de parâmetros
#   - mede métricas de teste (accuracy, f1, precision, recall, loss)
#   - salva tudo em treinamentos/estatisticas/<dataset>_<modelo>/fold_<k>/
#
# Depois:
# - agrega por modelo (média ± std entre folds)
# - escreve resumo.txt por modelo
# - escreve <dataset>_tabela_resumo.txt com todos os modelos
#
# IMPORTANTE:
# - NÃO roda o modelo "vector" (SSN puro). Ele será medido separadamente depois.
#
# Execução:
#   cd treinamentos
#   python extracao_all.py
#
# Estrutura esperada no repo:
# treinamentos/
#   config1.yaml
#   dataset.py
#   model.py
#   callbacks.py
#   estatisticas/   (pasta existe ou será criada)
#   extracao_all.py (este script)

import os
import time
import json
import yaml
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# importa implementações locais
from model import (
    CustomModel,
    CustomEnsembleModel,
    CustomModelTriple,
    CustomFeaturesOnlyModel,  # continua definido aqui mas NÃO será chamado nesse script
)
from dataset import (
    CustomImageModule_kf,
    CustomImageCSVModule_kf,
    # CustomFeaturesFromFoldersModule_kf será importado dinamicamente se/quando usado
)


# -------------------------------------------------
# utilidades básicas
# -------------------------------------------------

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
    Retorna o número total de amostras no dataloader de teste,
    usado pra calcular latência média e throughput.
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
    Para modelos imagem+vetor (ensemble / triple),
    inferimos o tamanho do vetor SSN a partir de um batch do datamodule.
    Esperado batch = (imgs, features, labels).
    """
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, (list, tuple)) and len(batch) >= 2, \
        "Batch inesperado. Esperava (img, features, label) para inferir features_dim."

    features = batch[1]
    if isinstance(features, (list, tuple)):
        features = torch.as_tensor(features[0])

    if features.ndim == 1:  # vetor 1D direto
        return int(features.shape[0])

    # caso usual: (B, dim)
    return int(features.shape[-1])


# -------------------------------------------------
# construção dinâmica do DataModule e do Modelo
# -------------------------------------------------

def build_datamodule(h, n_splits: int, fold_idx: int):
    """
    Cria o DataModule certo de acordo com o TMODEL.

    - convnext_t, swint_t:
        imagem apenas -> CustomImageModule_kf
    - convnext_t_ne, swint_t_ne, triple:
        imagem + vetor SSN -> CustomImageCSVModule_kf
    - vector:
        só vetor SSN -> CustomFeaturesFromFoldersModule_kf
        (não será rodado neste script, mas deixo a lógica preparada)
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
        # NÃO vamos chamar isso agora, mas mantemos pronto pro futuro.
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

    raise ValueError(f"TMODEL '{tmodel}' não reconhecido em build_datamodule().")


def build_model(h, dm_type, dm=None):
    """
    Cria a LightningModule correta conforme TMODEL.
    - convnext_t, swint_t           -> CustomModel
    - convnext_t_ne, swint_t_ne     -> CustomEnsembleModel (backbone + SSN)
    - triple                        -> CustomModelTriple   (ConvNeXt+Swin+SSN)
    - vector                        -> CustomFeaturesOnlyModel (só SSN)
      (vector não será usado aqui agora)
    """
    tmodel = h["TMODEL"]

    # normaliza optimizer_momentum (às vezes é float, às vezes lista)
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
        return CustomModel(
            tmodel=tmodel,
            **common_kwargs,
        )

    if tmodel in ["convnext_t_ne", "swint_t_ne"]:
        assert dm is not None, "Preciso do datamodule pra inferir features_dim."
        features_dim = infer_features_dim_from_dm(dm)
        # CustomEnsembleModel espera tmodel sem o sufixo "_ne"
        return CustomEnsembleModel(
            tmodel=tmodel.replace("_ne", ""),
            features_dim=features_dim,
            **common_kwargs,
        )

    if tmodel == "triple":
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
        # não vai ser chamado neste script, mas deixo pra referência futura
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

    raise ValueError(f"TMODEL '{tmodel}' não suportado em build_model().")


# -------------------------------------------------
# estatística e salvamento (agregação pós-fold)
# -------------------------------------------------

def mean_std(values):
    """Retorna (média, desvio_amostral) ignorando None."""
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return (None, None)
    if len(vals) == 1:
        return (float(vals[0]), 0.0)
    arr = np.array(vals, dtype=float)
    return (float(np.mean(arr)), float(np.std(arr, ddof=1)))


def aggregate_reports(reports, dataset_name, tmodel_name):
    """
    Recebe lista de 'report' (um por fold) e agrega:
    - tempos médios
    - uso de memória médio
    - accuracy média ± std
    - test_error média ± std  (1 - accuracy)
    - métricas macro médias
    """
    train_times = []
    test_times = []
    inf_ms = []
    throughput = []
    gpu_train = []
    gpu_test = []
    ckpt_mb = []
    nparams = []

    acc_list = []
    err_list = []
    f1_list = []
    prec_list = []
    rec_list = []
    loss_list = []

    for r in reports:
        train_times.append(r["train_time_sec"])
        test_times.append(r["test_time_sec"])
        inf_ms.append(r["test_inf_ms_per_sample"])
        throughput.append(r["throughput_samples_per_sec"])
        gpu_train.append(r["max_gpu_mem_mb_train"])
        gpu_test.append(r["max_gpu_mem_mb_test"])
        ckpt_mb.append(r["checkpoint_size_mb"])
        nparams.append(r["num_params_trainable"])

        tm = r["test_metrics"]
        # seu LightningModule deve logar essas métricas no test_step
        acc = tm.get("test_accuracy", None)
        f1  = tm.get("test_f1", None)
        prc = tm.get("test_precision", None)
        rec = tm.get("test_recall", None)
        los = tm.get("test_loss", None)

        if acc is not None:
            acc_list.append(acc)
            err_list.append(1.0 - acc)
        if f1 is not None:
            f1_list.append(f1)
        if prc is not None:
            prec_list.append(prc)
        if rec is not None:
            rec_list.append(rec)
        if los is not None:
            loss_list.append(los)

    agg = {
        "dataset": dataset_name,
        "tmodel": tmodel_name,
        "num_folds_ok": len(reports),

        # custo computacional
        "train_time_sec_mean_std": mean_std(train_times),
        "test_time_sec_mean_std": mean_std(test_times),
        "inf_ms_per_sample_mean_std": mean_std(inf_ms),
        "throughput_samples_per_sec_mean_std": mean_std(throughput),
        "max_gpu_mem_mb_train_mean_std": mean_std(gpu_train),
        "max_gpu_mem_mb_test_mean_std": mean_std(gpu_test),
        "checkpoint_size_mb_mean_std": mean_std(ckpt_mb),
        "num_params_trainable_mean_std": mean_std(nparams),

        # desempenho
        "test_accuracy_mean_std": mean_std(acc_list),
        "test_error_mean_std": mean_std(err_list),
        "test_f1_mean_std": mean_std(f1_list),
        "test_precision_mean_std": mean_std(prec_list),
        "test_recall_mean_std": mean_std(rec_list),
        "test_loss_mean_std": mean_std(loss_list),
    }
    return agg


def write_model_resumo(run_dir, agg):
    """
    Salva resumo.txt dentro de estatisticas/<dataset>_<tmodel>/,
    contendo médias ± std agregadas nos folds.
    """
    os.makedirs(run_dir, exist_ok=True)
    resumo_path = os.path.join(run_dir, "resumo.txt")

    def fmt_pair(pair):
        if pair is None or pair[0] is None:
            return "N/A"
        m, s = pair
        return f"{m:.6f} ± {s:.6f}"

    with open(resumo_path, "w") as f:
        f.write(f"=== RESUMO {agg['dataset']} / {agg['tmodel']} ===\n")
        f.write(f"Folds válidos: {agg['num_folds_ok']}\n\n")

        f.write("== Desempenho de teste ==\n")
        f.write(f"Acurácia (mean ± std): {fmt_pair(agg['test_accuracy_mean_std'])}\n")
        f.write(f"Test error (1-acc) (mean ± std): {fmt_pair(agg['test_error_mean_std'])}\n")
        f.write(f"F1 macro (mean ± std): {fmt_pair(agg['test_f1_mean_std'])}\n")
        f.write(f"Precision macro (mean ± std): {fmt_pair(agg['test_precision_mean_std'])}\n")
        f.write(f"Recall macro (mean ± std): {fmt_pair(agg['test_recall_mean_std'])}\n")
        f.write(f"Loss (mean ± std): {fmt_pair(agg['test_loss_mean_std'])}\n\n")

        f.write("== Custo computacional ==\n")
        f.write(f"Tempo treino s (mean ± std): {fmt_pair(agg['train_time_sec_mean_std'])}\n")
        f.write(f"Tempo teste s (mean ± std): {fmt_pair(agg['test_time_sec_mean_std'])}\n")
        f.write(f"Latência ms/img (mean ± std): {fmt_pair(agg['inf_ms_per_sample_mean_std'])}\n")
        f.write(f"Throughput img/s (mean ± std): {fmt_pair(agg['throughput_samples_per_sec_mean_std'])}\n")
        f.write(f"GPU train MB pico (mean ± std): {fmt_pair(agg['max_gpu_mem_mb_train_mean_std'])}\n")
        f.write(f"GPU test MB pico (mean ± std): {fmt_pair(agg['max_gpu_mem_mb_test_mean_std'])}\n")
        f.write(f"Tamanho ckpt MB (mean ± std): {fmt_pair(agg['checkpoint_size_mb_mean_std'])}\n")
        f.write(f"#params treináveis (mean ± std): {fmt_pair(agg['num_params_trainable_mean_std'])}\n")


def write_dataset_table(base_save_dir, dataset_name, dataset_results):
    """
    Gera um arquivo <dataset>_tabela_resumo.txt dentro de estatisticas/
    listando para cada tmodel (exceto vector, que não rodamos aqui):
        - Accuracy mean ± std
        - Test error mean ± std
        - Latência ms/img mean ± std
        - Tamanho ckpt MB mean ± std
    """
    tabela_path = os.path.join(base_save_dir, f"{dataset_name}_tabela_resumo.txt")

    # nomes mais bonitos pra aparecer na tabela final
    pretty_name = {
        "convnext_t":      "ConvNeXt",
        "convnext_t_ne":   "ConvNeXt+SSN",
        "swint_t":         "Swin",
        "swint_t_ne":      "Swin+SSN",
        "triple":          "Triple (C+S+SSN)",
        # "vector":        "SSN (vector)"  # propositalmente fora
    }

    def fmt_pair(pair):
        if pair is None or pair[0] is None:
            return "N/A"
        m, s = pair
        return f"{m:.6f} ± {s:.6f}"

    with open(tabela_path, "w") as f:
        f.write(f"TABELA RESUMO ({dataset_name})\n")
        f.write("Modelo ; Accuracy (mean±std) ; TestError (mean±std) ; Latência ms/img (mean±std) ; Ckpt MB (mean±std)\n")
        for tmodel, agg in dataset_results.items():
            # pula vector se ele estiver acidentalmente no dict
            if tmodel == "vector":
                continue
            f.write(
                f"{pretty_name.get(tmodel,tmodel)} ; "
                f"{fmt_pair(agg['test_accuracy_mean_std'])} ; "
                f"{fmt_pair(agg['test_error_mean_std'])} ; "
                f"{fmt_pair(agg['inf_ms_per_sample_mean_std'])} ; "
                f"{fmt_pair(agg['checkpoint_size_mb_mean_std'])}\n"
            )


# -------------------------------------------------
# núcleo: benchmark de UM modelo em UM fold (1 época)
# -------------------------------------------------

def benchmark_single_fold(h, fold_idx, base_save_dir, seed=42):
    """
    Executa benchmark de UM modelo (h['TMODEL']) em UM fold.
    - treina 1 época
    - mede custo
    - salva resultados brutos em estatisticas/<dataset>_<tmodel>/fold_<fold>/
    - retorna um dict 'report' com tudo
    """

    set_seeds(seed)

    dataset_name = h["NAME_DATASET"]
    tmodel_name = h["TMODEL"]

    run_name = f"{dataset_name}_{tmodel_name}"
    run_dir = os.path.join(base_save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # snapshot dos hiperparâmetros usados
    with open(os.path.join(fold_dir, "hyperparams_used.json"), "w") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

    # datamodule
    k_folds = int(h.get("K_FOLDS", 5))
    dm, dm_type = build_datamodule(h, n_splits=k_folds, fold_idx=fold_idx)
    dm.setup(stage="fit")

    # modelo lightning
    model = build_model(h, dm_type, dm=dm)

    # callback: salva best checkpoint por menor val_loss
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

    trainer = pl.Trainer(
        log_every_n_steps=10,
        accelerator=h["ACCELERATOR"],
        devices=h["DEVICES"],
        precision=h["PRECISION"],
        max_epochs=1,  # só 1 época para benchmark
        callbacks=callbacks,
    )

    # medir treino
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer.fit(model, dm)
    train_time_sec = time.perf_counter() - t0

    max_gpu_mb_train = None
    if torch.cuda.is_available():
        max_gpu_mb_train = torch.cuda.max_memory_allocated() / (1024**2)

    # info do best checkpoint
    best_model_path = ckpt_cb.best_model_path
    best_epoch = None
    try:
        ckpt_data = torch.load(best_model_path, map_location="cpu")
        best_epoch = ckpt_data.get("epoch", None)
    except Exception:
        pass

    # recarregar checkpoint pra medir validação/teste
    model = build_model(h, dm_type, dm=dm)
    model = model.load_from_checkpoint(best_model_path)

    val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]

    # teste com tempo de inferência
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

    # tamanho do checkpoint e nº de parâmetros
    checkpoint_size_mb = None
    if os.path.isfile(best_model_path):
        checkpoint_size_mb = os.path.getsize(best_model_path) / (1024**2)

    num_params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # monta o relatório bruto
    report = {
        "dataset": dataset_name,
        "tmodel": tmodel_name,
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

    # salva bruto em JSON
    with open(os.path.join(fold_dir, "benchmark_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # salva txt humano daquele fold
    with open(os.path.join(fold_dir, "benchmark_result.txt"), "w") as f:
        f.write("===== BENCHMARK REPORT (fold individual) =====\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print(f"[OK] {dataset_name}/{tmodel_name}/fold_{fold_idx} concluído.")
    return report


# -------------------------------------------------
# main: usa apenas config1.yaml e ignora "vector"
# -------------------------------------------------

def main():
    # diretório base = onde este script está
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_SAVE_DIR = os.path.join(BASE_DIR, "estatisticas")
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # carrega SOMENTE config1.yaml
    config_path = os.path.join(BASE_DIR, "config1.yaml")
    h_base = load_hparams(config_path)

    dataset_name = h_base["NAME_DATASET"]
    k_folds = int(h_base.get("K_FOLDS", 5))

    # modelos profundos que queremos comparar agora
    # (vector / SSN puro fica de fora de propósito)
    TMODELS_TO_TRY = [
        "convnext_t",
        "convnext_t_ne",
        "swint_t",
        "swint_t_ne",
        "triple",
    ]

    SEED = 42  # seed fixa p/ benchmark

    dataset_results = {}  # agregado por modelo

    # loop por modelo
    for tmodel_name in TMODELS_TO_TRY:
        print(f"\n--- Dataset {dataset_name} | Modelo {tmodel_name} ---")

        # clona config base e ajusta TMODEL
        h_local = dict(h_base)
        h_local["TMODEL"] = tmodel_name

        model_reports = []

        # roda todos os folds desse modelo
        for fold_idx in range(k_folds):
            try:
                rep = benchmark_single_fold(
                    h=h_local,
                    fold_idx=fold_idx,
                    base_save_dir=BASE_SAVE_DIR,
                    seed=SEED,
                )
                model_reports.append(rep)
            except Exception as e:
                print(f"[WARN] Falhou {dataset_name}/{tmodel_name}/fold_{fold_idx}: {e}")

        if not model_reports:
            print(f"[SKIP] Nenhum fold rodou para {dataset_name}/{tmodel_name}.")
            continue

        # agrega estatísticas entre folds
        agg = aggregate_reports(model_reports, dataset_name, tmodel_name)
        dataset_results[tmodel_name] = agg

        # salva resumo agregado por modelo
        run_name = f"{dataset_name}_{tmodel_name}"
        run_dir = os.path.join(BASE_SAVE_DIR, run_name)
        write_model_resumo(run_dir, agg)

    # salva tabela resumo geral do dataset
    write_dataset_table(BASE_SAVE_DIR, dataset_name, dataset_results)

    print("\n=== FINALIZADO. Resultados em treinamentos/estatisticas/ ===")


if __name__ == "__main__":
    main()
