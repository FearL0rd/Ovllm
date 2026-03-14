#!/usr/bin/env python3
"""
Example: Start Ovllm server programmatically.
"""

from ovllm.server import run_server
from ovllm.config import OvllmConfig


def main():
    """Start the Ovllm server."""
    config = OvllmConfig(
        host="0.0.0.0",
        port=11434,
    )

    print("Starting Ovllm server...")
    print(f"API: http://{config.host}:{config.port}")
    print("Press Ctrl+C to stop")

    run_server(config=config)


if __name__ == "__main__":
    main()
