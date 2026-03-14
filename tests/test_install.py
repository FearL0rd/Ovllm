#!/usr/bin/env python3
"""
Test script for Ovllm installation.

Run this to verify Ovllm is properly installed and configured.
"""

import sys


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from ovllm.config import OvllmConfig
        print("  [OK] ovllm.config")
    except ImportError as e:
        print(f"  [FAIL] ovllm.config: {e}")
        return False

    try:
        from ovllm.models import ModelManager
        print("  [OK] ovllm.models")
    except ImportError as e:
        print(f"  [FAIL] ovllm.models: {e}")
        return False

    try:
        from ovllm.engine import AsyncEngine, SamplingParams
        print("  [OK] ovllm.engine")
    except ImportError as e:
        print(f"  [FAIL] ovllm.engine: {e}")
        return False

    try:
        from ovllm.server import create_app
        print("  [OK] ovllm.server")
    except ImportError as e:
        print(f"  [FAIL] ovllm.server: {e}")
        return False

    try:
        from ovllm.cli.main import main
        print("  [OK] ovllm.cli.main")
    except ImportError as e:
        print(f"  [FAIL] ovllm.cli.main: {e}")
        return False

    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    from ovllm.config import OvllmConfig

    config = OvllmConfig()
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Models dir: {config.models_dir}")
    print(f"  GPU memory: {config.gpu_memory_utilization}")

    return True


def test_model_manager():
    """Test model manager initialization."""
    print("\nTesting model manager...")

    from ovllm.config import OvllmConfig
    from ovllm.models import ModelManager

    config = OvllmConfig()
    manager = ModelManager(config)

    models = manager.list_models()
    print(f"  Downloaded models: {len(models)}")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Ovllm Installation Test")
    print("=" * 50)

    if not test_imports():
        print("\n[FAIL] Import tests failed!")
        sys.exit(1)

    if not test_config():
        print("\n[FAIL] Configuration tests failed!")
        sys.exit(1)

    if not test_model_manager():
        print("\n[FAIL] Model manager tests failed!")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("[SUCCESS] All tests passed!")
    print("=" * 50)

    print("\nQuick start:")
    print("  1. Download a model: ovllm pull meta-llama/Llama-2-7b-chat-hf")
    print("  2. List models: ovllm list")
    print("  3. Run interactively: ovllm run meta-llama/Llama-2-7b-chat-hf")
    print("  4. Start server: ovllm serve")

    return 0


if __name__ == "__main__":
    sys.exit(main())
