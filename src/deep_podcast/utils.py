"""Utility & helper functions."""

from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from shared.config import get_system_config


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: Optional[str] = None) -> BaseChatModel:
    """Load a chat model from configuration or fully specified name.

    Args:
        fully_specified_name (str, optional): String in the format 'provider/model'.
                                             If not provided, uses configuration from .env
    """
    # Get system configuration
    config = get_system_config()
    
    # Use configuration from .env if no specific model is provided
    if fully_specified_name is None:
        provider = config.llm_config.provider
        model = config.llm_config.model
        api_key = config.llm_config.api_key
        base_url = config.llm_config.base_url
        temperature = config.llm_config.temperature
        max_tokens = config.llm_config.max_tokens
        
        # Initialize with configuration parameters
        return init_chat_model(
            model,
            model_provider=provider,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        # Fallback to the original behavior for backward compatibility
        provider, model = fully_specified_name.split("/", maxsplit=1)
        return init_chat_model(model, model_provider=provider)
