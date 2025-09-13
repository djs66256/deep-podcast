"""Utility functions for the Deep Research agent."""

import re
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

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


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with markdown
    text = re.sub(r'[^\w\s\-.,;:!?()[\]{}"\'/\\]', '', text)
    
    return text


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return "unknown"


def generate_filename(topic: str, extension: str = ".md") -> str:
    """Generate a filename for the research report."""
    # Create a hash of the topic for uniqueness
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
    
    # Clean topic for filename
    clean_topic = re.sub(r'[^\w\s-]', '', topic).strip()
    clean_topic = re.sub(r'[\s]+', '_', clean_topic)
    
    # Limit length
    if len(clean_topic) > 50:
        clean_topic = clean_topic[:50]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{timestamp}_{clean_topic}_{topic_hash}{extension}"


async def ensure_output_dir(base_dir: str) -> Path:
    """Ensure output directory exists and return Path object."""
    output_dir = Path(base_dir)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    return output_dir


def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def filter_content_by_length(content: str, min_length: int = 500) -> bool:
    """Check if content meets minimum length requirement."""
    return len(content.strip()) >= min_length


def deduplicate_urls(urls: List[str]) -> List[str]:
    """Remove duplicate URLs while preserving order."""
    seen = set()
    result = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources for inclusion in the research report."""
    formatted = []
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Unknown Title')
        url = source.get('url', '')
        domain = extract_domain(url)
        formatted.append(f"{i}. [{title}]({url}) - {domain}")
    
    return "\n".join(formatted)


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key phrases from text (simple implementation)."""
    # This is a basic implementation; in production, consider using NLP libraries
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    
    # Count word frequency
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top phrases
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_phrases]]