import os
from pathlib import Path
import shutil

# 📂 pasta onde estão os arquivos .ckpt
base_dir = Path("modelos_kf/D2_convnext_t")

# 🔍 lista todos os arquivos com sufixo -v
ckpt_files = sorted(base_dir.glob("fold_*_best_model-v*.ckpt"))

for file in ckpt_files:
    # cria o novo nome sem o sufixo -vN
    new_name = file.name.split("-v")[0] + ".ckpt"
    new_path = file.with_name(new_name)

    # ⚠️ Se já existir, verifica qual é o mais novo e mantém esse
    if new_path.exists():
        current_mtime = new_path.stat().st_mtime
        new_mtime = file.stat().st_mtime

        if new_mtime > current_mtime:
            print(f"📂 Substituindo {new_path.name} pelo mais recente {file.name}")
            shutil.move(str(file), str(new_path))
        else:
            print(f"🗑️ {file.name} é mais antigo que {new_path.name}, removendo...")
            file.unlink()
    else:
        print(f"✅ Renomeando {file.name} → {new_name}")
        shutil.move(str(file), str(new_path))

print("🎯 Renomeação concluída!")
