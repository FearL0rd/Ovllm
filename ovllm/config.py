"""
Configuration management for Ovllm.

Handles default settings, environment variables, and user preferences.
"""

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field


def _default_allow_patterns() -> List[str]:
    return ["*.safetensors", "*.json", "*.txt", "*.model"]


def _default_ignore_patterns() -> List[str]:
    return ["*.git", "*.gitattributes", "*.md", "README*", "LICENSE*"]


@dataclass
class OvllmConfig:
    """Configuration for Ovllm."""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 11434

    # Model storage
    models_dir: str = str(Path.home() / ".ovllm" / "models")

    # vLLM configuration
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None

    # HuggingFace configuration
    hf_token: Optional[str] = None

    # Download configuration
    allow_patterns: List[str] = field(default_factory=_default_allow_patterns)
    ignore_patterns: List[str] = field(default_factory=_default_ignore_patterns)

    # Logging
    log_level: str = "INFO"

    def __post_init__(self):
        """Apply environment variable overrides after initialization."""
        # Override with environment variables if set
        if os.getenv("OVLLM_HOST"):
            self.host = os.getenv("OVLLM_HOST", self.host)
        if os.getenv("OVLLM_PORT"):
            self.port = int(os.getenv("OVLLM_PORT", str(self.port)))
        if os.getenv("OVLLM_MODELS_DIR"):
            self.models_dir = os.getenv("OVLLM_MODELS_DIR", self.models_dir)
        if os.getenv("OVLLM_GPU_MEMORY"):
            self.gpu_memory_utilization = float(
                os.getenv("OVLLM_GPU_MEMORY", str(self.gpu_memory_utilization))
            )
        if os.getenv("HF_TOKEN"):
            self.hf_token = os.getenv("HF_TOKEN")
        if os.getenv("OVLLM_LOG_LEVEL"):
            self.log_level = os.getenv("OVLLM_LOG_LEVEL", self.log_level)

    @classmethod
    def from_env(cls) -> "OvllmConfig":
        """Create configuration from environment variables."""
        return cls()

    def to_vllm_args(self) -> dict:
        """Convert to vLLM EngineArgs format."""
        args = {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.max_model_len is not None:
            args["max_model_len"] = self.max_model_len
        if self.hf_token is not None:
            args["hf_token"] = self.hf_token
        return args


# Global default configuration
default_config = OvllmConfig()


def get_config() -> OvllmConfig:
    """Get the current configuration."""
    return default_config


def set_config(config: OvllmConfig) -> None:
    """Set the global configuration."""
    global default_config
    default_config = config
