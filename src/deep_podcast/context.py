"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=20,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    
    output_base_dir: str = field(
        default="./outputs",
        metadata={
            "description": "Base directory for saving all generated files."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                env_value = os.environ.get(f.name.upper(), f.default)
                # Convert to appropriate type based on field type
                if f.type == int or str(f.type) == "<class 'int'>" or str(f.type) == 'int':
                    try:
                        if isinstance(env_value, str):
                            env_value = int(env_value)
                    except (ValueError, TypeError):
                        env_value = f.default
                elif f.type == float or str(f.type) == "<class 'float'>" or str(f.type) == 'float':
                    try:
                        if isinstance(env_value, str):
                            env_value = float(env_value)
                    except (ValueError, TypeError):
                        env_value = f.default
                elif f.type == bool or str(f.type) == "<class 'bool'>" or str(f.type) == 'bool':
                    if isinstance(env_value, str):
                        env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                
                setattr(self, f.name, env_value)
