"""Define the state structures for the Deep Research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Any
from datetime import datetime

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class ResearchReport:
    """Research report data structure."""
    topic: str
    summary: str
    key_findings: List[str]
    sections: Dict[str, str]
    sources: List[str]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class InputState:
    """Defines the input state for the Deep Research agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    topic: str = ""
    """The research topic provided by the user."""


@dataclass
class State(InputState):
    """Represents the complete state of the Deep Research agent."""

    is_last_step: IsLastStep = field(default=False)
    """Indicates whether the current step is the last one."""

    # Research workflow states
    search_queries: List[str] = field(default_factory=list)
    """Generated search queries for the topic."""

    search_results: List[Dict[str, Any]] = field(default_factory=list)
    """Raw search results from various sources."""

    crawled_content: List[Dict[str, Any]] = field(default_factory=list)
    """Content extracted from web pages."""

    analyzed_content: Dict[str, Any] = field(default_factory=dict)
    """Structured and analyzed content."""

    report_sections: Dict[str, str] = field(default_factory=dict)
    """Different sections of the final report."""

    final_report: str = ""
    """The final markdown research report."""

    report_path: str = ""
    """Path to the saved report file."""

    # Error handling
    errors: List[str] = field(default_factory=list)
    """List of errors encountered during processing."""