#!/usr/bin/env python3
"""
Ovllm - Quick Start Script

Run this to quickly start using Ovllm with OpenWebUI.

Usage:
    python quickstart.py              # Start server only
    python quickstart.py --webui      # Start server + OpenWebUI
"""

import argparse
import subprocess
import sys
import os


def check_requirements():
    """Check if requirements are installed."""
    try:
        import vllm
        import huggingface_hub
        import fastapi
        import uvicorn
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return False


def start_server():
    """Start the Ovllm server."""
    from ovllm.server import run_server
    from ovllm.config import OvllmConfig

    config = OvllmConfig()
    print("=" * 60)
    print("  OVLLM SERVER")
    print("=" * 60)
    print(f"  API: http://{config.host}:{config.port}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    run_server(config=config)


def start_with_docker():
    """Start Ovllm + OpenWebUI using Docker Compose."""
    print("Starting Ovllm + OpenWebUI with Docker Compose...")

    try:
        subprocess.run([
            "docker-compose", "up", "-d"
        ], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

        print("=" * 60)
        print("  OVLLM + OPENWEBUI STARTED")
        print("=" * 60)
        print("  Ovllm API:   http://localhost:11434")
        print("  OpenWebUI:   http://localhost:3000")
        print("=" * 60)
        print("\n  Next steps:")
        print("  1. Open http://localhost:3000 in your browser")
        print("  2. Create an account or login")
        print("  3. Go to Settings > Connections")
        print("  4. Ensure Ollama URL is http://ovllm:11434")
        print("\n  To stop: docker-compose down")
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        print(f"Error starting Docker: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Docker Compose not found. Please install Docker Desktop.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ovllm Quick Start")
    parser.add_argument(
        "--webui",
        action="store_true",
        help="Start OpenWebUI alongside Ovllm (requires Docker)"
    )
    args = parser.parse_args()

    if args.webui:
        start_with_docker()
    else:
        if not check_requirements():
            sys.exit(1)
        start_server()


if __name__ == "__main__":
    main()
