import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for source_dir in (ROOT / "src", ROOT / "src" / "experiments"):
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)
