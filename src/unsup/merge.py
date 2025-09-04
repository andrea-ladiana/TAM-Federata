#!/usr/bin/env python3
"""Unisci tutti i file .py nella cartella corrente in codebase.txt.

Comportamento:
- cerca tutti i file con estensione .py nella cartella di lavoro corrente
- esclude se stesso (il file merge.py)
- ordina i file per nome
- scrive in `codebase.txt`, inserendo un separatore con il nome del file prima di ogni contenuto
"""
import os
from pathlib import Path


def main():
    here = Path.cwd()
    # Nome del file script stesso (basename) — utile se viene eseguito con percorso diverso
    try:
        script_name = Path(__file__).name
    except NameError:
        script_name = 'merge.py'

    py_files = [p for p in here.iterdir() if p.is_file() and p.suffix == '.py' and p.name != script_name]
    py_files.sort(key=lambda p: p.name)

    out_path = here / 'codebase.txt'
    with out_path.open('w', encoding='utf-8') as fout:
        for p in py_files:
            fout.write(f'# ---- {p.name} ----\n')
            try:
                with p.open('r', encoding='utf-8') as fin:
                    fout.write(fin.read())
            except Exception as e:
                fout.write(f'# ERRORE leggendo {p.name}: {e}\n')
            fout.write('\n\n')

    print(f'Wrote {len(py_files)} files into {out_path}')


if __name__ == '__main__':
    main()
