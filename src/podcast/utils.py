"""Utility functions for the Podcast agent."""

import re
import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

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


def estimate_reading_time(text: str, words_per_minute: int = 180) -> int:
    """Estimate reading time in seconds for given text."""
    if not text or not text.strip():
        return 5  # Minimum 5 seconds for empty text
    
    # For Chinese text, count characters instead of words
    # Chinese speakers read about 200-300 characters per minute
    chinese_char_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    
    if chinese_char_count > len(text) * 0.5:  # Mostly Chinese text
        # Chinese text: ~250 characters per minute
        time_minutes = chinese_char_count / 250
    else:
        # English text: count words
        word_count = len(text.split())
        time_minutes = word_count / words_per_minute
    
    # Calculate time in seconds, minimum 3 seconds
    return max(3, int(time_minutes * 60))


def clean_dialog_text(text: str) -> str:
    """Clean dialog text for TTS processing."""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
    
    # Clean up special characters
    text = re.sub(r'[#\*\-\+]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def parse_script_dialog(script: str) -> List[Dict[str, str]]:
    """Parse script content to extract dialog segments."""
    segments = []
    lines = script.split('\n')
    
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for speaker pattern: [Speaker]: content
        speaker_match = re.match(r'\[(.*?)\]:\s*(.*)', line)
        if speaker_match:
            # Save previous segment if exists
            if current_speaker and current_content:
                segments.append({
                    'speaker': current_speaker,
                    'content': ' '.join(current_content).strip()
                })
            
            # Start new segment
            current_speaker = speaker_match.group(1)
            current_content = [speaker_match.group(2)]
        else:
            # Continue current speaker's content
            if current_speaker and line:
                current_content.append(line)
    
    # Add final segment
    if current_speaker and current_content:
        segments.append({
            'speaker': current_speaker,
            'content': ' '.join(current_content).strip()
        })
    
    return segments


def generate_audio_filename(speaker: str, segment_id: int, extension: str = ".mp3") -> str:
    """Generate filename for audio segment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_speaker = re.sub(r'[^\w]', '_', speaker.lower())
    return f"{timestamp}_{clean_speaker}_{segment_id:03d}{extension}"


def generate_script_filename(topic: str, extension: str = ".md") -> str:
    """Generate filename for podcast script."""
    # Create hash for uniqueness
    topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
    
    # Clean topic for filename
    clean_topic = re.sub(r'[^\w\s-]', '', topic).strip()
    clean_topic = re.sub(r'[\s]+', '_', clean_topic)
    
    # Limit length
    if len(clean_topic) > 50:
        clean_topic = clean_topic[:50]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"podcast_{timestamp}_{clean_topic}_{topic_hash}{extension}"


async def ensure_output_dir(base_dir: str, subdir: str = "podcast") -> Path:
    """Ensure output directory exists."""
    output_dir = Path(base_dir) / subdir
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    return output_dir


def split_long_text(text: str, max_length: int = 300) -> List[str]:
    """Split long text into smaller chunks for TTS processing."""
    if len(text) <= max_length:
        return [text]
    
    # Try to split at sentence boundaries
    sentences = re.split(r'[.!?。！？]', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add sentence ending punctuation back
        if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
            sentence += '。'
        
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}分{remaining_seconds}秒"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}小时{remaining_minutes}分钟"


def extract_topic_from_report(report: str) -> str:
    """Extract main topic from research report."""
    lines = report.split('\n')
    
    # Look for title in markdown format
    for line in lines:
        if line.startswith('# '):
            title = line[2:].strip()
            # Remove " - 深度研究报告" suffix if present
            title = re.sub(r'\s*-\s*深度研究报告\s*$', '', title)
            return title
    
    # Fallback: use first meaningful line
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and len(line) > 10:
            return line[:100]  # Limit length
    
    return "Unknown Topic"


def validate_audio_config(config: Dict[str, Any]) -> bool:
    """Validate audio configuration parameters."""
    required_fields = ['rate', 'pitch', 'volume']
    
    for field in required_fields:
        if field not in config:
            return False
    
    # Validate ranges
    if not (0.5 <= config['rate'] <= 2.0):
        return False
    if not (-10 <= config['pitch'] <= 10):
        return False
    if not (0.1 <= config['volume'] <= 2.0):
        return False
    
    return True