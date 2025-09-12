"""Configuration and context for the Podcast agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Dict

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the Podcast agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the podcast agent's interactions."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for script generation."
        },
    )

    default_duration: int = field(
        default=1800,  # 30 minutes
        metadata={
            "description": "Default target duration for podcast in seconds."
        },
    )

    character_count: int = field(
        default=2,
        metadata={
            "description": "Number of characters in the podcast dialog."
        },
    )

    segment_max_length: int = field(
        default=300,
        metadata={
            "description": "Maximum length of each dialog segment in characters."
        },
    )

    # TTS Configuration
    tts_provider: str = field(
        default="qwen",
        metadata={
            "description": "TTS service provider to use."
        },
    )

    qwen_tts_api_key: str = field(
        default="",
        metadata={
            "description": "API key for Qwen TTS service."
        },
    )

    qwen_tts_base_url: str = field(
        default="https://dashscope.aliyuncs.com/api/v1/services/aigc/text2speech/synthesis",
        metadata={
            "description": "Base URL for Qwen TTS API."
        },
    )

    voice_models: Dict[str, str] = field(
        default_factory=lambda: {
            "host": "xiaoyun",
            "guest": "xiaogang"
        },
        metadata={
            "description": "Voice model mappings for different characters."
        },
    )

    speech_rate: float = field(
        default=1.0,
        metadata={
            "description": "Speech rate for TTS (0.5 to 2.0)."
        },
    )

    audio_format: str = field(
        default="mp3",
        metadata={
            "description": "Output audio format."
        },
    )

    sample_rate: int = field(
        default=16000,
        metadata={
            "description": "Audio sample rate in Hz."
        },
    )

    output_base_dir: str = field(
        default="./outputs",
        metadata={
            "description": "Base directory for saving podcast files."
        },
    )

    cleanup_temp: bool = field(
        default=True,
        metadata={
            "description": "Whether to cleanup temporary audio files after processing."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_value = os.environ.get(f.name.upper(), f.default)
                # Special handling for dict fields
                if f.name == "voice_models" and isinstance(env_value, str):
                    continue  # Keep default dict
                setattr(self, f.name, env_value)