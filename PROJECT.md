# Ovllm Project Structure

```
ovllm/
├── README.md                 # Documentation
├── LICENSE                   # Apache 2.0 License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml           # Modern Python project config
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose for Ovllm + OpenWebUI
├── quickstart.py            # Quick start script
│
├── ovllm/                   # Main package
│   ├── __init__.py          # Package init
│   ├── __main__.py          # Module entry point
│   ├── config.py            # Configuration management
│   ├── models.py            # Model download/management
│   ├── engine.py            # vLLM engine wrapper
│   ├── server.py            # FastAPI server (OpenAI/Ollama compatible)
│   │
│   └── cli/                 # CLI package
│       ├── __init__.py
│       ├── main.py          # CLI commands implementation
│       └── runner.py        # CLI entry point
│
├── examples/                # Example scripts
│   ├── __init__.py
│   ├── README.md            # Examples documentation
│   ├── serve.py             # Start server example
│   └── client.py            # Python client example
│
└── tests/                   # Test package
    ├── __init__.py
    └── test_install.py      # Installation verification
```

## File Descriptions

### Core Files

| File | Description |
|------|-------------|
| `ovllm/config.py` | Configuration with environment variable support |
| `ovllm/models.py` | Model download/management from HuggingFace |
| `ovllm/engine.py` | vLLM engine wrapper with async support |
| `ovllm/server.py` | FastAPI server with OpenAI/Ollama API compatibility |
| `ovllm/cli/main.py` | CLI commands: run, pull, serve, list, rm, show, ps |

### CLI Commands

```bash
ovllm run <model>              # Run model interactively
ovllm pull <model>             # Download model from HuggingFace
ovllm serve [--host] [--port]  # Start API server
ovllm list                     # List downloaded models
ovllm rm <model>               # Remove a model
ovllm show <model>             # Show model details
ovllm ps                       # Show running models
```

### API Endpoints

#### OpenAI-Compatible
- `POST /v1/chat/completions` - Chat completion
- `POST /v1/completions` - Text completion
- `GET /v1/models` - List models

#### Ollama-Compatible
- `POST /api/generate` - Generate completion
- `POST /api/chat` - Chat completion
- `POST /api/pull` - Pull model
- `GET /api/tags` - List models
- `DELETE /api/delete` - Delete model

### Usage Examples

#### Install
```bash
pip install -e .
```

#### Run interactively
```bash
ovllm run meta-llama/Llama-2-7b-chat-hf
```

#### Start server
```bash
ovllm serve
```

#### Docker (with OpenWebUI)
```bash
docker-compose up -d
```

### Configuration

Environment variables:
- `OVLLM_HOST` - Server host (default: 0.0.0.0)
- `OVLLM_PORT` - Server port (default: 11434)
- `OVLLM_MODELS_DIR` - Model storage (default: ~/.ovllm/models)
- `OVLLM_GPU_MEMORY` - GPU memory utilization (default: 0.9)
- `HF_TOKEN` - HuggingFace token for private models
