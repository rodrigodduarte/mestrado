#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agrega_vector_modelos_kf.py  (v2 - saída padronizada)
- Percorre modelos_kf/*_vector/seed_*/fold_*
- Converte resultados_fold.txt -> benchmark_result.txt (padrão interno)
- Para cada dataset *_vector, cria:
    /home/.../treinamentos/estatisticas/<dataset>_custo/<dataset>_desempenho.txt
  no formato humano (mean ± std) igual ao 'desempenhos_vector_todos.txt' mostrado.
- Gera o consolidate:
    /home/.../treinamentos/estatisticas/desempenhos_vector_todos.txt
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev, StatisticsError

# ========= Helpers numéricos =========

def _to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).strip())
        except Exception:
            return None

def mstd(vals):
    """Retorna (média, std) para lista de floats (ignorando None).
       Se lista vazia: retorna (None, None). Se 1 elemento: std=0.0."""
    clean = [v for v in map(_to_float, vals) if v is not None]
    if not clean:
        return (None, None)
    if len(clean) == 1:
        return (clean[0], 0.0)
    try:
        return (mean(clean), stdev(clean))
    except StatisticsError:
        return (mean(clean), 0.0)

def fmt(m, s, nd=6):
    """Formata 'm ± s' com nd casas; se None, retorna 'N/A'."""
    if m is None:
        return "N/A"
    return f"{m:.{nd}f} ± {s:.{nd}f}"

# ========= Parsing =========

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_resultados_fold(txt_path: Path) -> dict:
    data = {
        "Seed": None,
        "Fold": None,
        "best_epoch": None,
        "train_time_sec": None,
        "test_time_sec": None,
        "test_inf_ms_per_sample": None,
        "throughput_samples_per_sec": None,
        # Alguns logs têm apenas um pico de memória
        "max_gpu_mem_mb": None,
        # Se existirem separados, aproveitamos:
        "max_gpu_mem_mb_train": None,
        "max_gpu_mem_mb_test": None,
        # Tamanho do checkpoint (proxy p/ tamanho do modelo)
        "best_checkpoint_size_mb": None,
        # #params se aparecer em algum log
        "num_params_trainable": None,
        "val_metrics": {},
        "test_metrics": {},
        "balance_mode": None,
        "class_weights": None,
        "class_weights_injected": None,
    }
    if not txt_path.is_file():
        return data

    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "Seed":
                data["Seed"] = value
            elif key == "Fold":
                data["Fold"] = value
            elif key in {
                "best_epoch",
                "train_time_sec",
                "test_time_sec",
                "test_inf_ms_per_sample",
                "throughput_samples_per_sec",
                "max_gpu_mem_mb",
                "max_gpu_mem_mb_train",
                "max_gpu_mem_mb_test",
                "best_checkpoint_size_mb",
                "num_params_trainable",
            }:
                data[key] = safe_float(value)
            elif key == "val_metrics_json":
                try:
                    data["val_metrics"] = json.loads(value)
                except Exception:
                    m = re.search(r"\{.*\}$", line)
                    if m:
                        try:
                            data["val_metrics"] = json.loads(m.group(0))
                        except Exception:
                            pass
            elif key == "test_metrics_json":
                try:
                    data["test_metrics"] = json.loads(value)
                except Exception:
                    m = re.search(r"\{.*\}$", line)
                    if m:
                        try:
                            data["test_metrics"] = json.loads(m.group(0))
                        except Exception:
                            pass
            elif key == "balance_mode":
                data["balance_mode"] = value
            elif key == "class_weights":
                try:
                    data["class_weights"] = json.loads(value.replace("'", '"'))
                except Exception:
                    pass
            elif key == "class_weights_injected":
                data["class_weights_injected"] = value

    return data

def write_benchmark_result_txt(
    out_path: Path,
    dataset: str,
    tmodel: str,
    seed: str,
    fold_idx: str,
    info: dict,
    force: bool = False,
):
    if out_path.exists() and not force:
        return False

    lines = []
    lines.append("===== BENCHMARK REPORT (fold individual) =====")
    lines.append(f"dataset: {dataset}")
    lines.append(f"tmodel: {tmodel}")
    if seed is not None:
        lines.append(f"seed: {seed}")
    lines.append(f"fold_idx: {fold_idx if fold_idx is not None else 'N/A'}")
    lines.append(f"epoch_trained: {int(info.get('best_epoch') or 0)}")
    lines.append(f"train_time_sec: {info.get('train_time_sec')}")
    lines.append(f"test_time_sec: {info.get('test_time_sec')}")
    lines.append(f"test_inf_ms_per_sample: {info.get('test_inf_ms_per_sample')}")
    lines.append(f"throughput_samples_per_sec: {info.get('throughput_samples_per_sec')}")
    # Memória:
    if info.get("max_gpu_mem_mb_train") is not None:
        lines.append(f"max_gpu_mem_mb_train: {info.get('max_gpu_mem_mb_train')}")
    if info.get("max_gpu_mem_mb_test") is not None:
        lines.append(f"max_gpu_mem_mb_test: {info.get('max_gpu_mem_mb_test')}")
    if info.get("max_gpu_mem_mb") is not None:
        lines.append(f"max_gpu_mem_mb: {info.get('max_gpu_mem_mb')}")
    # Tamanho e #params:
    if info.get("best_checkpoint_size_mb") is not None:
        lines.append(f"best_checkpoint_size_mb: {info.get('best_checkpoint_size_mb')}")
    if info.get("num_params_trainable") is not None:
        lines.append(f"num_params_trainable: {int(info.get('num_params_trainable'))}")
    # Métricas
    vm = info.get("val_metrics") or {}
    tm = info.get("test_metrics") or {}
    if vm:
        lines.append(f"val_metrics_json: {json.dumps(vm, ensure_ascii=False)}")
    if tm:
        lines.append(f"test_metrics_json: {json.dumps(tm, ensure_ascii=False)}")
    # Outros
    if info.get("balance_mode") is not None:
        lines.append(f"balance_mode: {info.get('balance_mode')}")
    if info.get("class_weights") is not None:
        lines.append(f"class_weights: {json.dumps(info['class_weights'], ensure_ascii=False)}")
    if info.get("class_weights_injected") is not None:
        lines.append(f"class_weights_injected: {info.get('class_weights_injected')}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True

def gather_from_benchmark_result(txt_path: Path) -> dict:
    out = {
        "dataset": None,
        "tmodel": None,
        "seed": None,
        "fold_idx": None,
        "best_epoch": None,
        "train_time_sec": None,
        "test_time_sec": None,
        "test_inf_ms_per_sample": None,
        "throughput_samples_per_sec": None,
        "max_gpu_mem_mb_train": None,
        "max_gpu_mem_mb_test": None,
        "max_gpu_mem_mb": None,
        "best_checkpoint_size_mb": None,
        "num_params_trainable": None,
        "val_accuracy": None,
        "val_loss": None,
        "test_accuracy": None,
        "test_f1": None,
        "test_precision": None,
        "test_recall": None,
        "test_loss": None,
    }
    if not txt_path.is_file():
        return out

    vm, tm = {}, {}
    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "dataset":
                out["dataset"] = value
            elif key == "tmodel":
                out["tmodel"] = value
            elif key == "seed":
                out["seed"] = value
            elif key == "fold_idx":
                out["fold_idx"] = value
            elif key == "epoch_trained":
                out["best_epoch"] = _to_float(value)
            elif key in {
                "train_time_sec",
                "test_time_sec",
                "test_inf_ms_per_sample",
                "throughput_samples_per_sec",
                "max_gpu_mem_mb_train",
                "max_gpu_mem_mb_test",
                "max_gpu_mem_mb",
                "best_checkpoint_size_mb",
                "num_params_trainable",
            }:
                out[key] = _to_float(value)
            elif key == "val_metrics_json":
                try:
                    vm = json.loads(value)
                except Exception:
                    pass
            elif key == "test_metrics_json":
                try:
                    tm = json.loads(value)
                except Exception:
                    pass

    if isinstance(vm, dict):
        out["val_accuracy"] = _to_float(vm.get("val_accuracy"))
        out["val_loss"] = _to_float(vm.get("val_loss"))
    if isinstance(tm, dict):
        out["test_accuracy"] = _to_float(tm.get("test_accuracy"))
        out["test_f1"] = _to_float(tm.get("test_f1"))
        out["test_precision"] = _to_float(tm.get("test_precision"))
        out["test_recall"] = _to_float(tm.get("test_recall"))
        out["test_loss"] = _to_float(tm.get("test_loss"))

    return out

# ========= Escrita no formato “desejado” =========

SEPLONG = "-" * 72
SEPMED  = "-" * 64

def write_dataset_human_report(dst_path: Path, dataset: str, rows: list):
    """Escreve <dataset>_desempenho.txt no formato desejado."""
    # Coletas
    acc = [r["test_accuracy"] for r in rows if r.get("test_accuracy") is not None]
    err = [1.0 - a for a in acc] if acc else []
    f1  = [r["test_f1"] for r in rows if r.get("test_f1") is not None]
    pre = [r["test_precision"] for r in rows if r.get("test_precision") is not None]
    rec = [r["test_recall"] for r in rows if r.get("test_recall") is not None]
    los = [r["test_loss"] for r in rows if r.get("test_loss") is not None]

    trn = [r["train_time_sec"] for r in rows if r.get("train_time_sec") is not None]
    tst = [r["test_time_sec"] for r in rows if r.get("test_time_sec") is not None]
    lat = [r["test_inf_ms_per_sample"] for r in rows if r.get("test_inf_ms_per_sample") is not None]
    thr = [r["throughput_samples_per_sec"] for r in rows if r.get("throughput_samples_per_sec") is not None]

    gpu_tr = [r["max_gpu_mem_mb_train"] for r in rows if r.get("max_gpu_mem_mb_train") is not None]
    gpu_te = [r["max_gpu_mem_mb_test"] for r in rows if r.get("max_gpu_mem_mb_test") is not None]
    gpu_1  = [r["max_gpu_mem_mb"] for r in rows if r.get("max_gpu_mem_mb") is not None]

    ckpt = [r["best_checkpoint_size_mb"] for r in rows if r.get("best_checkpoint_size_mb") is not None]
    npar = [r["num_params_trainable"] for r in rows if r.get("num_params_trainable") is not None]

    m_acc, s_acc = mstd(acc)
    m_err, s_err = mstd(err)
    m_f1,  s_f1  = mstd(f1)
    m_pre, s_pre = mstd(pre)
    m_rec, s_rec = mstd(rec)
    m_los, s_los = mstd(los)

    m_trn, s_trn = mstd(trn)
    m_tst, s_tst = mstd(tst)
    m_lat, s_lat = mstd(lat)
    m_thr, s_thr = mstd(thr)

    m_gtr, s_gtr = mstd(gpu_tr)
    m_gte, s_gte = mstd(gpu_te)
    m_gpu, s_gpu = mstd(gpu_1)

    m_ckp, s_ckp = mstd(ckpt)
    m_npa, s_npa = mstd(npar)

    lines = []
    lines.append(f"=== {dataset} / vector ===")
    lines.append("== Desempenho de teste ==")
    lines.append(f"Acurácia (mean ± std): {fmt(m_acc, s_acc)}")
    lines.append(f"Test error (1-acc) (mean ± std): {fmt(m_err, s_err)}")
    lines.append(f"F1 macro (mean ± std): {fmt(m_f1, s_f1)}")
    lines.append(f"Precision macro (mean ± std): {fmt(m_pre, s_pre)}")
    lines.append(f"Recall macro (mean ± std): {fmt(m_rec, s_rec)}")
    lines.append(f"Loss (mean ± std): {fmt(m_los, s_los)}")
    lines.append("")
    lines.append(SEPMED)
    lines.append("")
    lines.append("== Custo computacional ==")
    lines.append(f"Tempo treino s (mean ± std): {fmt(m_trn, s_trn)}")
    lines.append(f"Tempo teste s (mean ± std): {fmt(m_tst, s_tst)}")
    lines.append(f"Latência ms/img (mean ± std): {fmt(m_lat, s_lat)}")
    lines.append(f"Throughput img/s (mean ± std): {fmt(m_thr, s_thr)}")

    # Memória: imprime condicionando ao que existe
    if m_gtr is not None:
        lines.append(f"GPU train MB pico (mean ± std): {fmt(m_gtr, s_gtr)}")
    if m_gte is not None:
        lines.append(f"GPU test MB pico (mean ± std): {fmt(m_gte, s_gte)}")
    if m_gtr is None and m_gte is None and m_gpu is not None:
        lines.append(f"GPU MB pico (mean ± std): {fmt(m_gpu, s_gpu)}")

    lines.append(f"Tamanho do modelo MB (mean ± std): {fmt(m_ckp, s_ckp)}")
    if m_npa is not None:
        lines.append(f"#params treináveis (mean ± std): {fmt(m_npa, s_npa)}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def append_to_all(all_path: Path, dataset: str, fonte_path: Path):
    """Adiciona bloco com cabeçalho + (fonte) + conteúdo do arquivo do dataset ao consolidado."""
    all_path.parent.mkdir(parents=True, exist_ok=True)
    existing = all_path.read_text(encoding="utf-8") if all_path.exists() else ""
    block = []

    block.append(f"=== {dataset} / vector ===")
    block.append(f"(fonte: {str(fonte_path)})")
    block.append("")
    # repete cabeçalho dataset e o conteúdo dele
    content = fonte_path.read_text(encoding="utf-8")
    block.append(content.strip())
    block.append("")
    block.append(SEPLONG)
    block.append("")

    with all_path.open("w", encoding="utf-8") as f:
        # se já tem algo, concatena preservando
        if existing.strip():
            f.write(existing.rstrip() + "\n")
        f.write("\n".join(block))

# ========= Main =========

def main():
    parser = argparse.ArgumentParser(description="Agrega resultados do modelo vector (modelos_kf) com saída padronizada.")
    parser.add_argument("--base", type=str, required=True,
                        help="Diretório base de modelos_kf (ex.: /home/rodrigoduarte/Documentos/projeto/modelos_kf)")
    parser.add_argument("--force", action="store_true",
                        help="Se definido, sobrescreve benchmark_result.txt existentes.")
    parser.add_argument("--out-all", type=str,
                        default="/home/rodrigoduarte/Documentos/projeto/treinamentos/estatisticas/desempenhos_vector_todos.txt",
                        help="Arquivo consolidado final no formato desejado.")
    parser.add_argument("--per-dataset-dir", type=str,
                        default="/home/rodrigoduarte/Documentos/projeto/treinamentos/estatisticas",
                        help="Diretório base onde criaremos <dataset>_custo/<dataset>_desempenho.txt")
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    out_all = Path(args.out_all).expanduser().resolve()
    per_base = Path(args.per_dataset_dir).expanduser().resolve()

    # Limpa consolidado (recria do zero)
    if out_all.exists():
        out_all.unlink()

    # Encontra datasets do modelo 'vector'
    dataset_dirs = sorted([p for p in base_dir.iterdir()
                           if p.is_dir() and p.name.endswith("_vector")])

    created = 0
    touched = 0

    for ds_dir in dataset_dirs:
        dataset_name = ds_dir.name.replace("_vector", "")
        tmodel = "vector"

        # Se existir seed_*, usa; senão, procura folds direto
        seed_dirs = [p for p in ds_dir.glob("seed_*") if p.is_dir()]
        if not seed_dirs:
            seed_dirs = [ds_dir]

        rows = []

        for sd in seed_dirs:
            seed = None
            m = re.search(r"seed_(\d+)", sd.name)
            if m:
                seed = m.group(1)

            for fold_dir in sorted(sd.glob("fold_*")):
                if not fold_dir.is_dir():
                    continue
                m2 = re.search(r"fold_(\d+)", fold_dir.name)
                fold_idx = m2.group(1) if m2 else None

                res_txt = fold_dir / "resultados_fold.txt"
                bench_txt = fold_dir / "benchmark_result.txt"

                info = parse_resultados_fold(res_txt)
                if info["Seed"] and not seed:
                    seed = str(info["Seed"])

                wrote = write_benchmark_result_txt(
                    bench_txt,
                    dataset=dataset_name,
                    tmodel=tmodel,
                    seed=seed,
                    fold_idx=fold_idx,
                    info=info,
                    force=args.force,
                )
                created += int(wrote)
                touched += 1

                row = gather_from_benchmark_result(bench_txt)
                rows.append(row)

        # === Para este dataset, gera o arquivo humano no formato desejado ===
        per_ds_dir = per_base / f"{dataset_name}_custo"
        per_ds_path = per_ds_dir / f"{dataset_name}_desempenho.txt"
        write_dataset_human_report(per_ds_path, dataset_name, rows)

        # Adiciona ao consolidado único, com a linha (fonte: ...)
        append_to_all(out_all, dataset_name, per_ds_path)

    print(f"[OK] benchmark_result.txt criados/atualizados: {created}/{touched}")
    print(f"[OK] Arquivo consolidado (formato desejado): {out_all}")

if __name__ == "__main__":
    main()
