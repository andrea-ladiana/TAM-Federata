import io
import glob
from pathlib import Path


def fix_keyword_eq(path: Path) -> bool:
    import re
    data = path.read_text(encoding='utf-8')
    orig = data
    # Replace Greek xi used as keyword (ξ= or ξ =)
    data = re.sub(r"\u03be\s*=", "xi=", data)
    if data != orig:
        path.write_text(data, encoding='utf-8')
        return True
    return False


def main():
    root = Path(__file__).resolve().parents[1]
    changed = []
    for fp in sorted((root / 'experiments').glob('exp_*.py')):
        if fix_keyword_eq(fp):
            changed.append(fp.name)
    print('Patched:', ', '.join(changed) if changed else '(none)')


if __name__ == '__main__':
    main()
