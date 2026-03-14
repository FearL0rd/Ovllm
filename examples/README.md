# Ovllm Examples

Example usage of Ovllm with Python.

## Quick Start

```bash
# Start the server
python examples/serve.py

# Or use the CLI
ovllm serve
```

## Python Client Example

```python
from openai import OpenAI

# Connect to Ovllm (OpenAI-compatible API)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## cURL Example

```bash
# List models
curl http://localhost:11434/v1/models

# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```
