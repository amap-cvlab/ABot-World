"""Compatibility entry point for the Iris inference benchmark."""
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.setdefault("PROJECT_ROOT", str(root))

    from abot_world.iris.benchmark_cli import main

    main()
