"""Shared utilities and helper functions."""

import re
import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# Optional imports with fallbacks
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """Generate a unique task ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"task_{timestamp}_{random_hash}"


def clean_filename(text: str, max_length: int = 50) -> str:
    """Clean text for use as filename."""
    # Remove invalid characters
    cleaned = re.sub(r'[^\w\s-]', '', text)
    
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Limit length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    
    return cleaned.strip('_')


def create_topic_hash(topic: str) -> str:
    """Create a short hash for a topic."""
    return hashlib.md5(topic.encode()).hexdigest()[:8]


def generate_timestamped_filename(
    prefix: str, 
    topic: str, 
    extension: str,
    include_hash: bool = True
) -> str:
    """Generate a timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_topic = clean_filename(topic)
    
    parts = [prefix, timestamp, clean_topic]
    
    if include_hash:
        topic_hash = create_topic_hash(topic)
        parts.append(topic_hash)
    
    filename = "_".join(parts) + extension
    return filename


async def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists asynchronously."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def save_text_file(content: str, file_path: Union[str, Path]) -> bool:
    """Save text content to file asynchronously."""
    try:
        file_path = Path(file_path)
        await ensure_directory(file_path.parent)
        
        if HAS_AIOFILES:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
        else:
            # Fallback to synchronous write
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        logger.info(f"Successfully saved file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        return False


async def load_text_file(file_path: Union[str, Path]) -> Optional[str]:
    """Load text content from file asynchronously."""
    try:
        if HAS_AIOFILES:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        else:
            # Fallback to synchronous read
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        return content
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return None


async def save_json_file(data: Any, file_path: Union[str, Path]) -> bool:
    """Save data as JSON file asynchronously."""
    try:
        file_path = Path(file_path)
        await ensure_directory(file_path.parent)
        
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return await save_text_file(content, file_path)
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False


async def load_json_file(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from JSON file asynchronously."""
    try:
        content = await load_text_file(file_path)
        if content:
            return json.loads(content)
        return None
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return None


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return "unknown"


def sanitize_text(text: str) -> str:
    """Sanitize text for processing."""
    if not text:
        return ""
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def estimate_reading_time(text: str, words_per_minute: int = 180) -> int:
    """Estimate reading time in seconds."""
    word_count = len(text.split())
    time_minutes = word_count / words_per_minute
    return int(time_minutes * 60)


def format_duration(seconds: int) -> str:
    """Format duration in human readable format."""
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


def parse_speaker_line(line: str) -> Optional[tuple[str, str]]:
    """Parse speaker line in format '[Speaker]: Content'."""
    match = re.match(r'\[(.*?)\]:\s*(.*)', line.strip())
    if match:
        return match.group(1), match.group(2)
    return None


def split_into_chunks(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end),
                text.rfind('。', start, end),
                text.rfind('！', start, end),
                text.rfind('？', start, end),
            )
            
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


async def retry_async(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """Retry async function with exponential backoff."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break
            
            wait_time = delay * (backoff_factor ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
    
    raise last_exception


def extract_key_points(text: str, max_points: int = 10) -> List[str]:
    """Extract key points from text (basic implementation)."""
    # Split into sentences
    sentences = re.split(r'[.!?。！？]', text)
    
    # Filter and clean sentences
    key_points = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence) < 200:  # Filter by length
            key_points.append(sentence)
    
    # Return top sentences (could be improved with NLP analysis)
    return key_points[:max_points]


def calculate_content_score(content: str) -> float:
    """Calculate a quality score for content (0-1)."""
    if not content:
        return 0.0
    
    score = 0.0
    
    # Length score (favor medium-length content)
    length = len(content)
    if 100 <= length <= 2000:
        score += 0.3
    elif 50 <= length < 100 or 2000 < length <= 5000:
        score += 0.2
    elif length > 5000:
        score += 0.1
    
    # Structure score (presence of sentences)
    sentences = len(re.split(r'[.!?。！？]', content))
    if sentences >= 3:
        score += 0.3
    elif sentences >= 2:
        score += 0.2
    
    # Diversity score (variety of words)
    words = set(re.findall(r'\b\w+\b', content.lower()))
    if len(words) > 50:
        score += 0.2
    elif len(words) > 20:
        score += 0.1
    
    # Information density score
    # Simple heuristic: presence of numbers, proper nouns, etc.
    has_numbers = bool(re.search(r'\d+', content))
    has_capitals = bool(re.search(r'[A-Z][a-z]', content))
    
    if has_numbers and has_capitals:
        score += 0.2
    elif has_numbers or has_capitals:
        score += 0.1
    
    return min(score, 1.0)


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step: int, description: str = ""):
        """Update progress."""
        self.current_step = step
        percentage = (step / self.total_steps) * 100
        
        elapsed = datetime.now() - self.start_time
        if step > 0:
            estimated_total = elapsed.total_seconds() * (self.total_steps / step)
            remaining = estimated_total - elapsed.total_seconds()
        else:
            remaining = 0
        
        logger.info(f"Progress: {percentage:.1f}% ({step}/{self.total_steps}) - {description}")
        
        return {
            "percentage": percentage,
            "current_step": step,
            "total_steps": self.total_steps,
            "estimated_remaining": int(remaining),
            "description": description
        }