"""Shared data models and utilities for the Deep Podcast system."""

__version__ = "0.1.0"

from .models import (
    ResearchReport,
    PodcastScript,
    Character,
    DialogSegment,
    SystemConfig,
    LLMConfig,
    TTSConfig,
    SearchConfig,
    OutputConfig,
)

__all__ = [
    "ResearchReport",
    "PodcastScript", 
    "Character",
    "DialogSegment",
    "SystemConfig",
    "LLMConfig",
    "TTSConfig", 
    "SearchConfig",
    "OutputConfig",
]