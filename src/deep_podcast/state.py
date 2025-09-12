"""Define the state structures for the Deep Podcast controller agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Any
from datetime import datetime

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

from shared.models import CompletePodcastResult, GenerationProgress, TaskStatus


@dataclass
class InputState:
    """Defines the input state for the Deep Podcast controller.
    
    This represents the user's request to generate a complete podcast from a research topic.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.
    """
    
    user_topic: str = ""
    """The research topic provided by the user for podcast generation."""


@dataclass
class State(InputState):
    """Represents the complete state of the Deep Podcast controller.
    
    This coordinates the execution of both deep_research and podcast sub-graphs.
    """

    is_last_step: IsLastStep = field(default=False)
    """Indicates whether the current step is the last one."""

    # Task management
    task_id: str = ""
    """Unique identifier for this podcast generation task."""
    
    progress: Optional[GenerationProgress] = None
    """Current progress of the podcast generation process."""
    
    # Research phase state
    research_status: TaskStatus = TaskStatus.PENDING
    """Status of the research phase."""
    
    research_report: str = ""
    """Generated research report content."""
    
    research_report_path: str = ""
    """Path to the saved research report file."""
    
    # Podcast generation phase state
    podcast_status: TaskStatus = TaskStatus.PENDING
    """Status of the podcast generation phase."""
    
    podcast_script: str = ""
    """Generated podcast script content."""
    
    podcast_script_path: str = ""
    """Path to the saved podcast script file."""
    
    podcast_audio_path: str = ""
    """Path to the generated podcast audio file."""
    
    # Output management
    output_directory: str = ""
    """Directory containing all generated files."""
    
    # Timing and metadata
    start_time: Optional[datetime] = None
    """When the podcast generation started."""
    
    completion_time: Optional[datetime] = None
    """When the podcast generation completed."""
    
    # Error handling
    error_message: str = ""
    """Error message if generation failed."""
    
    errors: list[str] = field(default_factory=list)
    """List of errors encountered during processing."""
    
    # Final result
    final_result: Optional[CompletePodcastResult] = None
    """The complete result of the podcast generation process."""
