"""Make the src-layout package importable during pytest collection.

This keeps local test runs independent from an explicitly configured
PYTHONPATH environment variable.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

src_path = str(SRC_DIRECTORY)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
