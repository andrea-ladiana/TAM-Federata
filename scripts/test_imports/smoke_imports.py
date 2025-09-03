import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import src.unsup  # noqa: F401

def check(module):
    try:
        __import__(module)
        print(f"[OK] import {module}")
    except Exception as e:
        print(f"[FAIL] import {module}: {e}")

mods = [
    'unsup.networks',
    'unsup.functions',
    'unsup.dynamics',
    'unsup.dynamics_single_mode',
]

for m in mods:
    check(m)

exp = ROOT / 'experiments'
for py in sorted(exp.glob('exp_*.py')):
    mod = f"experiments.{py.stem}"
    try:
        __import__(mod)
        print(f"[OK] import {mod}")
    except Exception as e:
        print(f"[WARN] import {mod}: {e}")
