"""
Model management for Ovllm.

Handles downloading, caching, and managing models from HuggingFace.
"""

import os
import json
import shutil
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from huggingface_hub import snapshot_download, HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError
from tqdm import tqdm

from .config import OvllmConfig
from .gguf_merge import auto_merge_gguf, is_gguf_split_file, find_gguf_splits


@dataclass
class ModelInfo:
    """Information about a downloaded model."""

    # Model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf")
    model_id: str

    # Local path
    path: str

    # Download date
    downloaded_at: str

    # Model metadata
    size_bytes: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    # Revision/commit
    revision: str = "main"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        """Create from dictionary."""
        return cls(**data)


class ModelManager:
    """
    Manages model downloads and local storage.

    Models are stored in ~/.ovllm/models/<model_id>/ with:
    - Model weights (safetensors, etc.)
    - config.json
    - tokenizer files
    - metadata.json (Ovllm-specific metadata)

    Supports GGUF models with quantization suffix (e.g., model:Q4_K_M).
    """

    METADATA_FILE = "ovllm_metadata.json"

    def __init__(self, config: Optional[OvllmConfig] = None):
        self.config = config or OvllmConfig()
        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._api = None

    def _parse_gguf_model(self, model_id: str) -> tuple:
        """
        Parse GGUF model ID to extract quantization suffix.

        Args:
            model_id: Model ID, optionally with :quant suffix
                      (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
                            or "unsloth/MiniMax-M2.5-GGUF:UD-Q4_K_XL")

        Returns:
            Tuple of (base_model_id, quantization_suffix) or (model_id, None)
        """
        # Check for quantization suffix after colon (e.g., :Q4_K_M or :UD-Q4_K_XL)
        if ":" in model_id:
            parts = model_id.rsplit(":", 1)
            if len(parts) == 2:
                potential_quant = parts[1]
                # Valid quantization patterns: Q*, UD-*, IQ*, etc.
                if potential_quant and ("-" in potential_quant or potential_quant.startswith("Q") or potential_quant.startswith("I")):
                    return parts[0], potential_quant  # base model, quantization
        return model_id, None

    @property
    def hf_api(self) -> HfApi:
        """Get HuggingFace API client."""
        if self._api is None:
            self._api = HfApi(token=self.config.hf_token)
        return self._api

    def _sanitize_model_id(self, model_id: str) -> str:
        """Sanitize model ID for filesystem."""
        # Replace / with -- and : with - for directory naming
        # This handles model IDs like "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
        return model_id.replace("/", "--").replace(":", "-")

    def _is_gguf_model(self, model_id: str) -> bool:
        """Check if model is a GGUF model."""
        return "gguf" in model_id.lower()

    def _merge_gguf_splits(self, model_path: Path, quant_suffix: str = None) -> None:
        """
        Find and merge GGUF split files in the model directory.

        Args:
            model_path: Directory containing GGUF files
            quant_suffix: Optional quantization suffix to filter by
        """
        print(f"_merge_gguf_splits called with model_path={model_path}, quant_suffix={quant_suffix}")

        # Find the actual directory containing GGUF files
        # Files might be in model_path directly or in a subdirectory (e.g., quant_suffix/)
        gguf_dirs = [model_path]

        # Check if there's a subdirectory for the quant_suffix
        if quant_suffix:
            quant_dir = model_path / quant_suffix
            if quant_dir.exists() and quant_dir.is_dir():
                # Check if this dir has GGUF files
                if list(quant_dir.glob("*.gguf")):
                    gguf_dirs.append(quant_dir)
                    print(f"  Also checking quant directory: {quant_dir}")

        # Find all GGUF split files (pattern: *-*-of-*.gguf)
        gguf_splits = []
        for gguf_dir in gguf_dirs:
            for f in gguf_dir.glob("*.gguf"):
                print(f"  Found GGUF file: {f}")
                if '-of-' in f.name:
                    if quant_suffix is None or quant_suffix in f.name:
                        gguf_splits.append(f)

        print(f"  Found {len(gguf_splits)} split GGUF files")

        if not gguf_splits:
            # Also check if there are any subdirectories with GGUF files
            for subdir in model_path.iterdir():
                if subdir.is_dir():
                    for f in subdir.glob("*.gguf"):
                        print(f"  Found GGUF file in subdir: {f}")
                        if '-of-' in f.name:
                            gguf_splits.append(f)
                    if gguf_splits:
                        print(f"  Found {len(gguf_splits)} split GGUF files in subdirectory")
                        break

        if not gguf_splits:
            return

        # Group by base pattern - use the directory of the first file as reference
        groups = {}
        for f in gguf_splits:
            # Extract base name and part number
            # Pattern: name-00001-of-00004.gguf
            match = re.search(r'(.+?)-(\d+)-of-(\d+)\.gguf$', f.name)
            if match:
                base_name = match.group(1)
                part_num = int(match.group(2))
                total_parts = int(match.group(3))
                key = f"{base_name}-{total_parts}"
                if key not in groups:
                    groups[key] = []
                groups[key].append((part_num, f))

        # Merge each group
        for key, files in groups.items():
            if len(files) < 2:
                continue

            # Sort by part number
            files.sort(key=lambda x: x[0])
            sorted_files = [f for _, f in files]

            # Get the base name without part numbers
            first_name = sorted_files[0].name
            merged_name = re.sub(r'-\d+-of-\d+', '', first_name)

            # Use the directory of the first file (in case files are in a subdir)
            merged_path = sorted_files[0].parent / merged_name
            print(f"Merging {len(sorted_files)} GGUF parts into {merged_path}...")

            # Merge the files
            self._merge_gguf_files(sorted_files, merged_path)

            # Delete split files
            for f in sorted_files:
                f.unlink()
                print(f"  Deleted: {f}")

            # If merged file is in a subdirectory, move it to the main model path
            if merged_path.parent != model_path:
                final_path = model_path / merged_name
                print(f"  Moving merged file to {final_path}...")
                shutil.move(str(merged_path), str(final_path))

                # Clean up empty subdirectory
                if merged_path.parent != model_path and merged_path.parent.exists():
                    try:
                        merged_path.parent.rmdir()
                        print(f"  Removed empty directory: {merged_path.parent}")
                    except OSError:
                        pass  # Directory not empty, leave it

    def _merge_gguf_files(self, input_files: List[Path], output_file: Path) -> None:
        """
        Merge multiple GGUF files into a single file.

        Args:
            input_files: List of input GGUF files in order
            output_file: Path to output merged file
        """
        with open(output_file, 'wb') as out_f:
            for file_path in input_files:
                print(f"  Processing {file_path.name}...")
                with open(file_path, 'rb') as f:
                    out_f.write(f.read())

        print(f"Merged file created: {output_file}")

    def _get_model_path(self, model_id: str) -> Path:
        """Get local path for a model."""
        safe_id = self._sanitize_model_id(model_id)
        return self.models_dir / safe_id

    def _get_metadata_path(self, model_id: str) -> Path:
        """Get metadata file path for a model."""
        return self._get_model_path(model_id) / self.METADATA_FILE

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded."""
        model_path = self._get_model_path(model_id)
        return model_path.exists() and model_path.is_dir()

    def download(
        self,
        model_id: str,
        revision: str = "main",
        force: bool = False,
    ) -> ModelInfo:
        """
        Download a model from HuggingFace.

        Args:
            model_id: HuggingFace model ID, optionally with GGUF quant suffix
                      (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
            revision: Branch or commit hash
            force: Force re-download even if exists

        Returns:
            ModelInfo with download details
        """
        # Parse GGUF model with quantization suffix
        base_model_id, quant_suffix = self._parse_gguf_model(model_id)
        is_gguf = self._is_gguf_model(model_id) or "gguf" in model_id.lower()

        # For GGUF models, the model_id already contains the quant suffix
        # (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
        # We store and retrieve using the full model_id as provided
        store_model_id = model_id

        model_path = self._get_model_path(store_model_id)

        # Check if already downloaded
        if self.is_downloaded(store_model_id) and not force:
            return self.get_info(store_model_id)

        # Remove existing if force
        if force and model_path.exists():
            shutil.rmtree(model_path)

        model_path.mkdir(parents=True, exist_ok=True)

        if quant_suffix:
            print(f"Downloading GGUF model {base_model_id} ({quant_suffix})...")
        else:
            print(f"Downloading {model_id} ({revision})...")

        # Download with progress
        try:
            if is_gguf and quant_suffix:
                # Download specific GGUF file(s) for quantization
                print(f"Searching for GGUF files with quantization {quant_suffix}...")

                # Get the list of files in the repo - use base_model_id (without :quant suffix)
                files = list_repo_files(repo_id=base_model_id, revision=revision)

                # Find all matching GGUF files for this quantization
                gguf_files = []
                for file_path in files:
                    if file_path.endswith(".gguf"):
                        if quant_suffix in file_path:
                            gguf_files.append(file_path)

                if not gguf_files:
                    # Fallback: try to find any GGUF files
                    for file_path in files:
                        if file_path.endswith(".gguf"):
                            gguf_files.append(file_path)

                if gguf_files:
                    # Sort files to ensure consistent ordering
                    gguf_files.sort()
                    print(f"Downloading {len(gguf_files)} GGUF file(s): {', '.join(gguf_files)}")

                    # Download all GGUF files
                    for gguf_file in gguf_files:
                        print(f"  Downloading {gguf_file}...")
                        hf_hub_download(
                            repo_id=base_model_id,  # Use base_model_id without :quant suffix
                            filename=gguf_file,
                            revision=revision,
                            local_dir=str(model_path),
                            local_dir_use_symlinks=False,
                            token=self.config.hf_token,
                        )
                else:
                    raise ValueError(f"No GGUF files found with quantization {quant_suffix}")
            elif is_gguf:
                # GGUF model without specific quantization - download all GGUF files
                print(f"Downloading GGUF model files...")
                snapshot_download(
                    repo_id=base_model_id,  # Use base_model_id without :quant suffix
                    revision=revision,
                    local_dir=str(model_path),
                    token=self.config.hf_token,
                    allow_patterns=["*.gguf", "*.json"],
                )
            else:
                # Regular model download
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_dir=str(model_path),
                    token=self.config.hf_token,
                    ignore_patterns=self.config.ignore_patterns,
                )
        except RepositoryNotFoundError:
            if model_path.exists():
                shutil.rmtree(model_path)
            raise ValueError(
                f"Model '{model_id}' not found on HuggingFace Hub. "
                "Please check the model ID or your authentication."
            )

        # For GGUF models, check if we need to merge split files
        print(f"Checking for GGUF split files (is_gguf={is_gguf})...")
        if is_gguf:
            self._merge_gguf_splits(model_path, quant_suffix)

        # Calculate size
        size_bytes = sum(
            f.stat().st_size for f in model_path.glob("**/*") if f.is_file()
        )

        # Load model config if available
        model_config = {}
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                model_config = json.load(f)

        # Save Ovllm metadata
        metadata = ModelInfo(
            model_id=store_model_id,
            path=str(model_path),
            downloaded_at=datetime.now().isoformat(),
            size_bytes=size_bytes,
            config=model_config,
            revision=revision,
        )

        metadata_path = self._get_metadata_path(store_model_id)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(f"Model downloaded to {model_path}")
        return metadata

    def get_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a downloaded model."""
        # Try sanitized path first
        metadata_path = self._get_metadata_path(model_id)

        if not metadata_path.exists():
            # Try to reconstruct from existing directory
            model_path = self._get_model_path(model_id)
            if model_path.exists():
                size_bytes = sum(
                    f.stat().st_size
                    for f in model_path.glob("**/*")
                    if f.is_file()
                )
                return ModelInfo(
                    model_id=model_id,
                    path=str(model_path),
                    downloaded_at=datetime.now().isoformat(),
                    size_bytes=size_bytes,
                )
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return ModelInfo.from_dict(data)

    def list_models(self) -> List[ModelInfo]:
        """List all downloaded models."""
        models = []

        for metadata_file in self.models_dir.glob(f"**/{self.METADATA_FILE}"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                models.append(ModelInfo.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue

        return models

    def remove(self, model_id: str) -> bool:
        """
        Remove a downloaded model.

        Returns:
            True if removed, False if not found
        """
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            return False

        shutil.rmtree(model_path)
        return True

    def get_model_path(self, model_id: str) -> str:
        """Get the local path for a model."""
        model_path = self._get_model_path(model_id)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model '{model_id}' not found. "
                f"Run 'ovllm pull {model_id}' to download it."
            )

        return str(model_path)

    def get_or_download(self, model_id: str, revision: str = "main") -> str:
        """
        Get model path, downloading if necessary.

        Returns:
            Local path to the model
        """
        if not self.is_downloaded(model_id):
            self.download(model_id, revision)

        return self.get_model_path(model_id)
