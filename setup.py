"""
PyPI package configuration for Ovllm.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ovllm",
    version="0.1.0",
    author="Ovllm Team",
    description="Ollama-like CLI for vLLM - download and serve HuggingFace models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "vllm>=0.6.0",
        "huggingface_hub>=0.20.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "tqdm>=4.66.0",
        "openai>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ovllm=ovllm.cli.main:main",
        ],
    },
)
