#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agrega_vector_modelos_kf.py
- Passa por modelos_kf/*_vector/seed_*/fold_*/
- Converte resultados_fold.txt -> benchmark_result.txt (padrão já utilizado)
- Gera um arquivo único com tudo do modelo vector:
  /home/rodrigoduarte/Documentos/projeto/treinamentos/estatisticas/vector_agregado_desempenho.txt
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

# ========= Helpers =========

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_resultados_fold(txt_path: Path) -> dict:
    """
    Lê resultados_fold.txt e retorna um dicionário com os campos relevantes.
    Aceita linhas no formato 'chave: valor' e JSON nas chaves *_json e listas.
    """
    data = {
        "Seed": None,
        "Fold": None,
        "best_epoch": None,
        "train_time_sec": None,
        "test_time_sec": None,
        "test_inf_ms_per_sample": None,
        "throughput_samples_per_sec": None,
        "max_gpu_mem_mb": None,
        "best_checkpoint_size_mb": None,
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
                "best_checkpoint_size_mb",
            }:
                data[key] = safe_float(value)
            elif key == "val_metrics_json":
                # value é JSON após o ":", então já está em 'value'
                try:
                    data["val_metrics"] = json.loads(value)
                except Exception:
                    # tenta achar JSON usando regex (caso tenha comentários à direita)
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
                # lista no formato [ ... ]
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
    """
    Escreve benchmark_result.txt no padrão já usado.
    Se existir e force=False, não sobrescreve.
    """
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
    # No vector, só há 'max_gpu_mem_mb' (um valor). Mantemos esse campo.
    if info.get("max_gpu_mem_mb") is not None:
        lines.append(f"max_gpu_mem_mb: {info.get('max_gpu_mem_mb')}")
    # Tamanho do checkpoint (equivalente a model_size do convnext)
    if info.get("best_checkpoint_size_mb") is not None:
        lines.append(f"best_checkpoint_size_mb: {info.get('best_checkpoint_size_mb')}")
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
    """
    Lê benchmark_result.txt e extrai campos básicos para o agregadão final.
    """
    out = {
        "dataset": None,
        "tmodel": None,
        "seed": None,
        "fold_idx": None,
        "best_epoch": None,
        "val_accuracy": None,
        "val_loss": None,
        "test_accuracy": None,
        "test_f1": None,
        "test_precision": None,
        "test_recall": None,
        "test_loss": None,
        "test_inf_ms_per_sample": None,
        "throughput_samples_per_sec": None,
        "max_gpu_mem_mb": None,
        "best_checkpoint_size_mb": None,
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

            if key in out:
                out[key] = value
            elif key == "dataset":
                out["dataset"] = value
            elif key == "tmodel":
                out["tmodel"] = value
            elif key == "seed":
                out["seed"] = value
            elif key == "fold_idx":
                out["fold_idx"] = value
            elif key == "epoch_trained":
                out["best_epoch"] = value
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
            elif key == "test_inf_ms_per_sample":
                out["test_inf_ms_per_sample"] = value
            elif key == "throughput_samples_per_sec":
                out["throughput_samples_per_sec"] = value
            elif key == "max_gpu_mem_mb":
                out["max_gpu_mem_mb"] = value
            elif key == "best_checkpoint_size_mb":
                out["best_checkpoint_size_mb"] = value

    # Acopla métricas
    if isinstance(vm, dict):
        out["val_accuracy"] = vm.get("val_accuracy")
        out["val_loss"] = vm.get("val_loss")
    if isinstance(tm, dict):
        out["test_accuracy"] = tm.get("test_accuracy")
        out["test_f1"] = tm.get("test_f1")
        out["test_precision"] = tm.get("test_precision")
        out["test_recall"] = tm.get("test_recall")
        out["test_loss"] = tm.get("test_loss")

    return out

# ========= Main =========

def main():
    parser = argparse.ArgumentParser(description="Agrega resultados do modelo vector (modelos_kf).")
    parser.add_argument("--base", type=str, required=True,
                        help="Diretório base de modelos_kf (ex.: /home/rodrigoduarte/Documentos/projeto/modelos_kf)")
    parser.add_argument("--force", action="store_true",
                        help="Se definido, sobrescreve benchmark_result.txt mesmo que exista.")
    parser.add_argument("--out", type=str, default="/home/rodrigoduarte/Documentos/projeto/treinamentos/estatisticas/vector_agregado_desempenho.txt",
                        help="Caminho do arquivo agregado final do modelo vector.")
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Encontrar datasets do modelo 'vector'
    dataset_dirs = sorted([p for p in base_dir.iterdir()
                           if p.is_dir() and p.name.endswith("_vector")])

    created = 0
    touched = 0
    aggregate_rows = []

    for ds_dir in dataset_dirs:
        dataset_name = ds_dir.name.replace("_vector", "")
        tmodel = "vector"

        # seeds (pode haver seed_42, seed_123, etc.). Se não houver, procurar folds direto.
        seed_dirs = [p for p in ds_dir.glob("seed_*") if p.is_dir()]
        if not seed_dirs:
            seed_dirs = [ds_dir]  # trata como se não houvesse subnível de seed

        for sd in seed_dirs:
            seed = None
            m = re.search(r"seed_(\d+)", sd.name)
            if m:
                seed = m.group(1)

            # folds
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

                # Para o agregadão final
                row = gather_from_benchmark_result(bench_txt)
                aggregate_rows.append(row)

    # Gera arquivo único do modelo vector
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "===== VECTOR AGREGADO (modelos_kf) =====",
        f"gerado_em: {now}",
        f"datasets_vector_encontrados: {len(dataset_dirs)}",
        f"folds_processados: {len(aggregate_rows)}",
        "",
        "# Campos: dataset | seed | fold | best_epoch | val_acc | val_loss | test_acc | test_f1 | test_precision | test_recall | test_loss | inf_ms | thr_sps | gpu_mem_mb | ckpt_mb",
        "",
    ]
    lines = header[:]
    for r in aggregate_rows:
        line = (
            f"{r.get('dataset')} | {r.get('seed')} | {r.get('fold_idx')} | "
            f"{r.get('best_epoch')} | {r.get('val_accuracy')} | {r.get('val_loss')} | "
            f"{r.get('test_accuracy')} | {r.get('test_f1')} | {r.get('test_precision')} | {r.get('test_recall')} | {r.get('test_loss')} | "
            f"{r.get('test_inf_ms_per_sample')} | {r.get('throughput_samples_per_sec')} | {r.get('max_gpu_mem_mb')} | {r.get('best_checkpoint_size_mb')}"
        )
        lines.append(line)

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] benchmark_result.txt criados/atualizados: {created}/{touched}")
    print(f"[OK] Agregado do modelo vector: {out_file}")

if __name__ == "__main__":
    main()
