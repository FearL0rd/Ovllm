"""
vLLM Engine wrapper for Ovllm.

Provides a simple interface to vLLM's LLM engine.
"""

import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator, List, Dict, Any
from dataclasses import dataclass

from vllm import LLM
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.outputs import RequestOutput

from .config import OvllmConfig
from .models import ModelManager


@dataclass
class SamplingParams:
    """Sampling parameters for generation."""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 256
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    seed: Optional[int] = None

    def to_vllm(self) -> VLLMSamplingParams:
        """Convert to vLLM SamplingParams."""
        return VLLMSamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            n=self.n,
            seed=self.seed,
        )


class Engine:
    """
    Wrapper around vLLM's LLM engine.

    Manages model loading and generation with a simple API.
    """

    def __init__(
        self,
        config: Optional[OvllmConfig] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        self.config = config or OvllmConfig()
        self.model_manager = model_manager or ModelManager(config)
        self._llm: Optional[LLM] = None
        self._current_model: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._llm is not None

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model ID."""
        return self._current_model

    def load_model(
        self,
        model_id: str,
        revision: str = "main",
        **kwargs,
    ) -> None:
        """
        Load a model into the engine.

        Args:
            model_id: HuggingFace model ID, optionally with GGUF quant suffix
                      (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
            revision: Model revision
            **kwargs: Additional vLLM arguments
        """
        # Get or download model
        model_path = self.model_manager.get_or_download(model_id, revision)
        model_path_obj = Path(model_path)

        # Check for GGUF model
        gguf_file = self._find_gguf_file(model_path_obj)

        # Merge configuration
        vllm_args = self.config.to_vllm_args()
        vllm_args.update(kwargs)

        if gguf_file:
            print(f"Loading GGUF model {gguf_file.name}...")
            # For GGUF models, point directly to the .gguf file
            self._llm = LLM(
                model=str(gguf_file),
                tokenizer=str(gguf_file),
                revision=revision,
                trust_remote_code=True,
                **vllm_args,
            )
        else:
            print(f"Loading model {model_id} from {model_path}...")
            self._llm = LLM(
                model=model_path,
                tokenizer=model_path,
                revision=revision,
                trust_remote_code=True,
                **vllm_args,
            )
        self._current_model = model_id

        print(f"Model {model_id} loaded successfully!")

    def _find_gguf_file(self, model_path: Path) -> Optional[Path]:
        """Find the GGUF file in the model directory.

        Priority:
        1. Merged GGUF file (no -of- pattern)
        2. Largest GGUF file (likely merged)
        3. First GGUF file found
        """
        try:
            gguf_files = list(model_path.glob("*.gguf"))
            if not gguf_files:
                return None

            # First, try to find a non-split GGUF file (merged file)
            for f in gguf_files:
                if '-of-' not in f.name:
                    return f

            # If all are split files, return the first one
            # (the merge should have happened during download)
            return gguf_files[0]
        except Exception:
            pass
        return None

    def unload_model(self) -> None:
        """Unload the current model."""
        self._llm = None
        self._current_model = None

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters

        Returns:
            List of generated completions
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if sampling_params is None:
            sampling_params = SamplingParams()

        vllm_params = sampling_params.to_vllm()
        outputs = self._llm.generate(prompts, vllm_params)

        return [output.outputs[0].text for output in outputs]

    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncIterator[str]:
        """
        Generate completion with streaming tokens.

        Args:
            prompt: Input prompt
            sampling_params: Generation parameters

        Yields:
            Generated tokens as they are produced
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if sampling_params is None:
            sampling_params = SamplingParams()

        vllm_params = sampling_params.to_vllm()

        # Note: vLLM's streaming requires async setup
        # For now, use sync generation
        outputs = self._llm.generate([prompt], vllm_params)

        if outputs:
            yield outputs[0].outputs[0].text


class AsyncEngine:
    """
    Async wrapper for vLLM engine with streaming support.
    """

    def __init__(
        self,
        config: Optional[OvllmConfig] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        self.config = config or OvllmConfig()
        self.model_manager = model_manager or ModelManager(config)
        self._llm: Optional[LLM] = None
        self._current_model: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    def load_model(
        self,
        model_id: str,
        revision: str = "main",
        **kwargs,
    ) -> None:
        """Load model synchronously."""
        model_path = self.model_manager.get_or_download(model_id, revision)
        model_path_obj = Path(model_path)

        # Check for GGUF model
        gguf_file = self._find_gguf_file(model_path_obj)

        vllm_args = self.config.to_vllm_args()
        vllm_args.update(kwargs)

        if gguf_file:
            print(f"Loading GGUF model {gguf_file.name}...")
            self._llm = LLM(
                model=str(gguf_file),
                tokenizer=str(gguf_file),
                revision=revision,
                trust_remote_code=True,
                **vllm_args,
            )
        else:
            print(f"Loading model {model_id} from {model_path}...")
            self._llm = LLM(
                model=model_path,
                tokenizer=model_path,
                revision=revision,
                trust_remote_code=True,
                **vllm_args,
            )
        self._current_model = model_id

        print(f"Model {model_id} loaded successfully!")

    def _find_gguf_file(self, model_path: Path) -> Optional[Path]:
        """Find the GGUF file in the model directory."""
        try:
            gguf_files = list(model_path.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]
        except Exception:
            pass
        return None

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncIterator[str]:
        """
        Generate with streaming output.

        Args:
            prompt: User prompt (for completion API)
            system_prompt: System prompt (for chat API)
            messages: Chat messages (for chat API)
            sampling_params: Generation parameters

        Yields:
            Generated text tokens
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded.")

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Build full prompt for chat
        if messages:
            # Format as chat
            full_prompt = self._format_chat(messages, system_prompt)
        else:
            full_prompt = prompt

        # Run sync generation in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llm.generate(
                [full_prompt],
                sampling_params.to_vllm(),
            ),
        )

        if result:
            yield result[0].outputs[0].text

    def _format_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format messages as chat prompt."""
        formatted = ""

        if system_prompt:
            formatted += f"System: {system_prompt}\n\n"

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"

        formatted += "Assistant:"
        return formatted
