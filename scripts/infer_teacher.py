"""Compatibility entry point for Iris BiT."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abot_world.bit.cli import main, read_conditions, read_example

if __name__ == "__main__":
    main()
