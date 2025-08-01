import os
from pathlib import Path

# Caminho da pasta onde estão os modelos
base_dir = Path("modelos_kf/D2_convnext_t")

# Lista todos os arquivos .ckpt
ckpt_files = sorted(base_dir.glob("fold_*_best_model*.ckpt"))

# Agrupar arquivos por fold
fold_files = {}
for file in ckpt_files:
    # Pega o nome base do fold (ex: fold_0_best_model)
    fold_key = "_".join(file.stem.split("_")[:3])  # fold_X_best_model
    if fold_key not in fold_files:
        fold_files[fold_key] = []
    fold_files[fold_key].append(file)

# Para cada fold, manter só o mais novo e apagar os outros
for fold, files in fold_files.items():
    # Ordena os arquivos pela data de modificação (do mais novo para o mais antigo)
    files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

    # Mantém o mais recente
    latest = files_sorted[0]
    print(f"[{fold}] Mantendo: {latest.name}")

    # Apaga os outros
    for old_file in files_sorted[1:]:
        print(f"   ➜ Excluindo: {old_file.name}")
        old_file.unlink()

print("✅ Limpeza concluída! Apenas o checkpoint mais recente de cada fold foi mantido.")
