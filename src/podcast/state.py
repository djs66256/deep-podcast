"""Define the state structures for the Podcast agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any
from datetime import datetime

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

from shared.models import Character, DialogSegment, PodcastScript


@dataclass
class InputState:
    """Defines the input state for the Podcast agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    input_report: str = ""
    """The research report to convert into a podcast."""


@dataclass
class State(InputState):
    """Represents the complete state of the Podcast agent."""

    is_last_step: IsLastStep = field(default=False)
    """Indicates whether the current step is the last one."""

    # Content analysis
    key_points: List[Dict[str, Any]] = field(default_factory=list)
    """Key points extracted from the research report."""

    dialog_structure: Dict[str, Any] = field(default_factory=dict)
    """Structure and flow of the podcast dialog."""

    characters: List[Character] = field(default_factory=list)
    """Character definitions for the podcast."""

    # Script generation
    script_content: str = ""
    """Generated podcast script in markdown format."""

    script_path: str = ""
    """Path to the saved script file."""

    # Audio generation
    audio_segments: List[Dict[str, Any]] = field(default_factory=list)
    """Information about generated audio segments."""

    final_audio_path: str = ""
    """Path to the final combined audio file."""

    # Metadata
    estimated_duration: int = 0
    """Estimated duration in seconds."""

    # Error handling
    errors: List[str] = field(default_factory=list)
    """List of errors encountered during processing."""