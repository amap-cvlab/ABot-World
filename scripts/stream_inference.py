"""Compatibility entry point for Iris streaming."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abot_world.iris.stream_cli import main

if __name__ == "__main__":
    main()
