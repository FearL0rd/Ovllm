"""
Ovllm - Run as module: python -m ovllm
"""

import sys
from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())
