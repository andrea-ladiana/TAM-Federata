# -*- coding: utf-8 -*-
"""
Utility I/O generiche per Exp-06 (single-only).

- ensure_dir(path): crea directory (parents=True, exist_ok=True)
- read_json/write_json
- save/load NPY/NPZ
- list_round_dirs: restituisce le cartelle round_XXX ordinate
- find_files: glob semplice (con opzione ricorsiva)
- atomic_write: scrittura robusta su file (rinomina atomica)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import tempfile
import shutil
import fnmatch


# ---------------------------------------------------------------------
# Directory & file helpers
# ---------------------------------------------------------------------
def ensure_dir(p: str | os.PathLike) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path: str | os.PathLike, data: bytes, mode: str = "wb") -> None:
    """
    Scrive dati su file in modo atomico: scrive su un file temporaneo
    nella stessa directory e poi rinomina.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, mode) as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        # ripulisce in caso di errori
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------
def read_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    data = json.dumps(obj, indent=2).encode("utf-8")
    atomic_write(path, data, mode="wb")


# ---------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------
def save_npy(path: str | os.PathLike, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: str | os.PathLike) -> np.ndarray:
    return np.load(path)


def save_npz(path: str | os.PathLike, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path: str | os.PathLike) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------------
# Scansione run/round
# ---------------------------------------------------------------------
def list_round_dirs(run_dir: str | os.PathLike) -> List[Path]:
    """
    Restituisce le cartelle "round_XXX" presenti in run_dir, ordinate per XXX.
    """
    rdir = Path(run_dir)
    if not rdir.exists():
        return []
    dirs = [p for p in rdir.iterdir() if p.is_dir() and p.name.startswith("round_")]
    # ordina per indice numerico
    def _key(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 10**9
    return sorted(dirs, key=_key)


def find_files(root: str | os.PathLike, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """
    Trova file in 'root' che matchano il pattern (glob a-la fnmatch). Se recursive=True,
    scende nelle sottodirectory.
    """
    root = Path(root)
    if not root.exists():
        return []
    matches: List[Path] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if fnmatch.fnmatch(name, pattern):
                    matches.append(Path(dirpath) / name)
    else:
        for p in root.iterdir():
            if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                matches.append(p)
    return matches
