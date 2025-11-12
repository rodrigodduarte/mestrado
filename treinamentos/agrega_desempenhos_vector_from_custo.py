#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import sys
from datetime import datetime

HEADER_RE = re.compile(r"^===\s*(?P<dataset>.+?)\s*/\s*(?P<modelo>.+?)\s*===\s*$")

def find_treinamentos_from_cwd() -> Path | None:
    """Procura um diretório 'treinamentos/estatisticas' subindo a partir do CWD."""
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        t = p / "treinamentos"
        if (t / "estatisticas").is_dir():
            return t
    return None

def split_blocks(text: str):
    """Divide o conteúdo em blocos iniciados por '=== dataset / modelo ==='."""
    lines = text.splitlines()
    blocks, cur_header, cur_lines = [], None, []
    for ln in lines:
        m = HEADER_RE.match(ln.strip())
        if m:
            if cur_header is not None:
                blocks.append((cur_header, "\n".join(cur_lines).strip()))
            cur_header = (m.group("dataset").strip(), m.group("modelo").strip())
            cur_lines = [ln]  # mantém o cabeçalho no bloco
        else:
            if cur_header is not None:
                cur_lines.append(ln)
    if cur_header is not None:
        blocks.append((cur_header, "\n".join(cur_lines).strip()))
    return blocks

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        T = find_treinamentos_from_cwd()
        if T is None:
            print("ERRO: não encontrei 'treinamentos/estatisticas' a partir do diretório atual.\n"
                  "Use: python agrega_desempenhos_vector_from_custo.py /caminho/para/treinamentos [saida.txt]",
                  file=sys.stderr)
            sys.exit(2)
        out_name = "desempenhos_vector_todos.txt"
    elif len(args) == 1:
        T = Path(args[0]).expanduser().resolve()
        out_name = "desempenhos_vector_todos.txt"
    else:
        T = Path(args[0]).expanduser().resolve()
        out_name = args[1]

    EST = T / "estatisticas"
    if not EST.is_dir():
        print(f"ERRO: diretório não encontrado: {EST}", file=sys.stderr)
        sys.exit(2)

    OUT = EST / out_name
    if OUT.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        OUT = EST / f"{OUT.stem}-{ts}{OUT.suffix}"

    agreg, fontes = [], []
    count_blocks = 0

    # Varre treinamentos/estatisticas/*_custo/*_desempenho.txt
    for d in sorted(p for p in EST.iterdir() if p.is_dir() and p.name.endswith("_custo")):
        for f in sorted(d.glob("*_desempenho.txt")):
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[AVISO] Falha lendo {f}: {e}", file=sys.stderr)
                continue

            for (dataset, modelo), bloco in split_blocks(txt):
                if "vector" in modelo.lower():
                    header = f"=== {dataset} / {modelo} ===\n(fonte: {f})\n"
                    agreg.append(header + "\n" + bloco)
                    fontes.append(str(f))
                    count_blocks += 1

    if count_blocks == 0:
        print("Nenhum bloco de desempenho com modelo '*vector' encontrado em treinamentos/estatisticas/*_custo/*_desempenho.txt",
              file=sys.stderr)
        sys.exit(1)

    sep = "\n\n" + ("-" * 72) + "\n\n"
    OUT.write_text(sep.join(agreg) + "\n", encoding="utf-8")

    # Manifesto das fontes usadas
    (EST / "desempenhos_vector_manifest.txt").write_text(
        "\n".join(sorted(set(fontes))) + "\n", encoding="utf-8"
    )

    print(f"[OK] Gerado: {OUT}")
    print(f"[OK] Blocos (vector) mesclados: {count_blocks}")
    print(f"[OK] Manifest: {(EST / 'desempenhos_vector_manifest.txt')}")

if __name__ == "__main__":
    main()
