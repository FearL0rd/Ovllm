#!/usr/bin/env python3
"""
Example: Use Ovllm with Python client.
"""

from openai import OpenAI


def main():
    # Initialize client (OpenAI-compatible)
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
    )

    # Chat completion
    print("Sending chat request...")

    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    print("\nResponse:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
