import os
import sys
from pathlib import Path


def project_root() -> Path:
    """Return repository root (parent of experiments/)."""
    return Path(__file__).resolve().parents[1]


def ensure_src_on_path() -> Path:
    """Prepend <root>/src to sys.path if missing and return it."""
    root = project_root()
    src = root / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def silence_tf(level: str = '3') -> None:
    """Set environment variables to silence TensorFlow logs (optional)."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', level)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

