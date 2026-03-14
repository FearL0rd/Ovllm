#!/usr/bin/env python3
"""
Ovllm CLI entry point.

Run with: python -m ovllm or ovllm (after pip install)
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
