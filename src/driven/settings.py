import os
from pathlib import Path

# Package paths
BASE_DIR = Path(os.path.dirname(os.path.relpath(__file__)))
PROJECT_PATH = BASE_DIR.resolve()

# Project root (ada repository root)
PROJECT_ROOT = PROJECT_PATH.parent.parent.parent

# Agent extensions (bundled with package)
EXTENSIONS_PATH = PROJECT_PATH / "extensions"

