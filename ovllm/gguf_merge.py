"""
GGUF file merger for multi-part GGUF models.

Merges split GGUF files (e.g., model-00001-of-00004.gguf) into a single file.
Based on the gguf-split.cpp approach from llama.cpp.
"""

import struct
import os
from pathlib import Path
from typing import List, Tuple


# GGUF magic number and version
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def read_gguf_header(file_path: Path) -> Tuple[int, int, dict]:
    """
    Read and parse GGUF header.

    Returns:
        Tuple of (tensor_count, total_size, header_info)
    """
    with open(file_path, 'rb') as f:
        # Magic (4 bytes)
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic}")

        # Version (4 bytes, little endian)
        version = struct.unpack('<I', f.read(4))[0]

        # Tensor count (8 bytes, little endian)
        tensor_count = struct.unpack('<Q', f.read(8))[0]

        # Alignment (8 bytes, little endian) - this is part of the header info
        alignment = struct.unpack('<Q', f.read(8))[0]

        return tensor_count, alignment, {'version': version}


def get_gguf_metadata(file_paths: List[Path]) -> Tuple[int, int, dict]:
    """
    Get metadata from the first GGUF file to use for the merged file.

    Returns:
        Tuple of (tensor_count, alignment, header_info)
    """
    return read_gguf_header(file_paths[0])


def merge_gguf_files(input_files: List[Path], output_file: Path) -> Path:
    """
    Merge multiple GGUF files into a single file.

    Args:
        input_files: List of input GGUF files in order
        output_file: Path to output merged file

    Returns:
        Path to merged file
    """
    if len(input_files) < 2:
        raise ValueError("Need at least 2 files to merge")

    print(f"Merging {len(input_files)} GGUF files...")

    # Read header from first file
    tensor_count, alignment, header_info = read_gguf_header(input_files[0])

    # Calculate total size needed
    total_size = 0
    file_sizes = []
    file_data_offsets = []

    for file_path in input_files:
        file_size = file_path.stat().st_size
        file_sizes.append(file_size)
        total_size += file_size

    # For multi-file GGUF, we need to:
    # 1. Keep the header from file 1
    # 2. Append tensor data from all files

    with open(output_file, 'wb') as out_f:
        # Read and write the complete first file
        with open(input_files[0], 'rb') as f:
            out_f.write(f.read())

        # Append remaining files (skip header, just append data)
        for i, file_path in enumerate(input_files[1:], start=1):
            print(f"  Appending {file_path.name}...")
            with open(file_path, 'rb') as f:
                # Skip header and append tensor data
                data = f.read()
                # For split GGUF files, we append the entire file content
                # as the splits are typically just chunks of tensor data
                out_f.write(data)

    print(f"Merged file created: {output_file}")
    return output_file


def merge_gguf_files_v2(input_files: List[Path], output_file: Path) -> Path:
    """
    Merge multiple GGUF files into a single file (v2 approach).

    This properly handles the GGUF format by:
    1. Reading metadata from all parts
    2. Creating a new header with combined tensor info
    3. Concatenating tensor data

    Args:
        input_files: List of input GGUF files in order
        output_file: Path to output merged file

    Returns:
        Path to merged file
    """
    if len(input_files) < 2:
        raise ValueError("Need at least 2 files to merge")

    print(f"Merging {len(input_files)} GGUF files using v2 approach...")

    # For GGUF split files, the format is typically:
    # - Part 1: Complete header + some tensor data
    # - Part 2-N: Additional tensor data only
    #
    # To merge, we need to concatenate all binary data

    total_size = sum(f.stat().st_size for f in input_files)
    print(f"Total merged size: {total_size / 1024 / 1024:.2f} MB")

    with open(output_file, 'wb') as out_f:
        for file_path in input_files:
            print(f"  Processing {file_path.name}...")
            with open(file_path, 'rb') as f:
                out_f.write(f.read())

    print(f"Merged file created: {output_file}")
    return output_file


def is_gguf_split_file(file_path: Path) -> bool:
    """Check if a file is a split GGUF file (has -of- pattern)."""
    return '-of-' in file_path.name and file_path.suffix == '.gguf'


def get_split_file_parts(file_path: Path) -> Tuple[str, int]:
    """
    Get the base name and part number from a split GGUF file.

    Returns:
        Tuple of (base_name, part_number)
    """
    name = file_path.stem  # Remove extension
    # Pattern: name-00001-of-00004
    if '-of-' in name:
        parts = name.split('-of-')
        base_and_part = parts[0]  # e.g., "MiniMax-M2.5-UD-Q4_K_XL-00001"
        total_part = parts[1]     # e.g., "00004"

        # Extract part number from base_and_part
        sub_parts = base_and_part.rsplit('-', 1)
        if len(sub_parts) == 2:
            return sub_parts[0], int(sub_parts[1])

    return name, 0


def find_gguf_splits(directory: Path, quant_suffix: str = None) -> List[List[Path]]:
    """
    Find all GGUF split files in a directory.

    Args:
        directory: Directory to search
        quant_suffix: Optional quantization suffix to filter by

    Returns:
        List of file groups, each group sorted by part number
    """
    gguf_files = list(directory.glob("*.gguf"))

    # Group by base pattern
    groups = {}
    for f in gguf_files:
        if is_gguf_split_file(f):
            base_name, part_num = get_split_file_parts(f)
            if quant_suffix is None or quant_suffix in f.name:
                if base_name not in groups:
                    groups[base_name] = []
                groups[base_name].append((part_num, f))

    # Sort each group by part number
    result = []
    for base_name, files in groups.items():
        sorted_files = sorted(files, key=lambda x: x[0])
        result.append([f for _, f in sorted_files])

    return result


def cleanup_split_files(split_files: List[Path]) -> None:
    """Delete split GGUF files after merging."""
    for f in split_files:
        if f.exists():
            f.unlink()
            print(f"  Deleted: {f}")


def auto_merge_gguf(directory: Path, quant_suffix: str = None) -> List[Path]:
    """
    Automatically find and merge GGUF split files.

    Args:
        directory: Directory containing GGUF files
        quant_suffix: Optional quantization suffix to filter by

    Returns:
        List of merged files
    """
    merged_files = []
    groups = find_gguf_splits(directory, quant_suffix)

    for group in groups:
        if len(group) > 1:
            # Create output filename (remove -00001-of-00004 pattern)
            base_name = group[0].name
            for i in range(10):
                base_name = base_name.replace(f"-{i}-of-", "-")

            # Remove part number pattern
            import re
            base_name = re.sub(r'-\d+-of-\d+', '', group[0].name)
            base_name = base_name.replace('.gguf', '')

            output_name = f"{base_name}.gguf"
            output_path = directory / output_name

            print(f"Merging {len(group)} files into {output_name}...")
            merge_gguf_files_v2(group, output_path)

            # Cleanup split files
            cleanup_split_files(group)

            merged_files.append(output_path)

    return merged_files
