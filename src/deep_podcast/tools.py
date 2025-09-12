"""Tools for the Deep Podcast controller agent.

Note: The Deep Podcast controller primarily coordinates sub-graphs
rather than using individual tools. Most functionality is handled
by the deep_research and podcast sub-graphs.
"""

from typing import Any, Callable, List

# Minimal tool set for the controller
# Most work is delegated to sub-graphs
TOOLS: List[Callable[..., Any]] = []
