"""Data models for the Deep Podcast system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum


class ResearchDepth(Enum):
    """Research depth levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class PodcastStyle(Enum):
    """Podcast style options."""
    CONVERSATIONAL = "conversational"
    INTERVIEW = "interview"
    NARRATIVE = "narrative"
    EDUCATIONAL = "educational"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


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
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "topic": self.topic,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "sections": self.sections,
            "sources": self.sources,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "file_path": self.file_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResearchReport:
        """Create from dictionary representation."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Character:
    """Character/Role definition for podcast."""
    name: str
    role: str
    personality: str
    voice_config: Dict[str, Any]
    background: Optional[str] = None
    expertise: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "role": self.role,
            "personality": self.personality,
            "voice_config": self.voice_config,
            "background": self.background,
            "expertise": self.expertise or [],
        }


@dataclass
class DialogSegment:
    """A segment of dialog in the podcast."""
    segment_id: int
    speaker: str
    content: str
    emotion: str
    duration_estimate: int  # in seconds
    audio_file: Optional[str] = None
    voice_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "content": self.content,
            "emotion": self.emotion,
            "duration_estimate": self.duration_estimate,
            "audio_file": self.audio_file,
            "voice_config": self.voice_config,
        }


@dataclass
class PodcastScript:
    """Complete podcast script data structure."""
    title: str
    characters: List[Character]
    segments: List[DialogSegment]
    total_duration: int  # in seconds
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    script_path: Optional[str] = None
    audio_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "characters": [char.to_dict() for char in self.characters],
            "segments": [seg.to_dict() for seg in self.segments],
            "total_duration": self.total_duration,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "script_path": self.script_path,
            "audio_path": self.audio_path,
        }


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60


@dataclass
class TTSConfig:
    """TTS configuration."""
    provider: str = "qwen"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    voice_models: Dict[str, str] = field(default_factory=lambda: {
        "host": "xiaoyun",
        "guest": "xiaogang"
    })
    audio_format: str = "mp3"
    sample_rate: int = 16000
    speech_rate: float = 1.0
    pitch: int = 0
    volume: float = 1.0


@dataclass
class SearchConfig:
    """Search and crawling configuration."""
    max_search_results: int = 20
    max_crawl_pages: int = 50
    search_timeout: int = 30
    crawl_timeout: int = 15
    content_min_length: int = 500
    concurrent_requests: int = 5


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "./outputs"
    report_format: str = "markdown"
    audio_format: str = "mp3"
    audio_quality: str = "high"
    file_naming: str = "{timestamp}_{topic_hash}"
    compression: bool = True
    cleanup_temp: bool = True
    
    def get_output_dir(self, subdir: str = "") -> Path:
        """Get output directory path."""
        if subdir:
            return Path(self.base_dir) / subdir
        return Path(self.base_dir)


@dataclass
class SystemConfig:
    """Complete system configuration."""
    llm_config: LLMConfig
    tts_config: TTSConfig
    search_config: SearchConfig
    output_config: OutputConfig
    
    @classmethod
    def from_env(cls) -> SystemConfig:
        """Create configuration from environment variables."""
        import os
        
        llm_config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620"),
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
        )
        
        tts_config = TTSConfig(
            api_key=os.getenv("QWEN_TTS_API_KEY", ""),
            base_url=os.getenv("QWEN_TTS_BASE_URL", TTSConfig.base_url),
            speech_rate=float(os.getenv("TTS_SPEECH_RATE", "1.0")),
            pitch=int(os.getenv("TTS_PITCH", "0")),
            volume=float(os.getenv("TTS_VOLUME", "1.0")),
        )
        
        search_config = SearchConfig(
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "20")),
            max_crawl_pages=int(os.getenv("MAX_CRAWL_PAGES", "50")),
            search_timeout=int(os.getenv("SEARCH_TIMEOUT", "30")),
            crawl_timeout=int(os.getenv("CRAWL_TIMEOUT", "15")),
            content_min_length=int(os.getenv("CONTENT_MIN_LENGTH", "500")),
        )
        
        output_config = OutputConfig(
            base_dir=os.getenv("OUTPUT_BASE_DIR", "./outputs"),
            cleanup_temp=os.getenv("CLEANUP_TEMP", "true").lower() == "true",
        )
        
        return cls(
            llm_config=llm_config,
            tts_config=tts_config,
            search_config=search_config,
            output_config=output_config,
        )


@dataclass
class ResearchResult:
    """Result of research operation."""
    status: TaskStatus
    report: Optional[ResearchReport] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class PodcastResult:
    """Result of podcast generation operation."""
    status: TaskStatus
    script: Optional[PodcastScript] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class CompletePodcastResult:
    """Result of complete podcast generation from topic."""
    status: TaskStatus
    research_result: Optional[ResearchResult] = None
    podcast_result: Optional[PodcastResult] = None
    error_message: Optional[str] = None
    total_processing_time: Optional[float] = None
    
    @property
    def report_path(self) -> Optional[str]:
        """Get research report file path."""
        if self.research_result and self.research_result.report:
            return self.research_result.report.file_path
        return None
    
    @property
    def script_path(self) -> Optional[str]:
        """Get podcast script file path."""
        if self.podcast_result and self.podcast_result.script:
            return self.podcast_result.script.script_path
        return None
    
    @property
    def audio_path(self) -> Optional[str]:
        """Get podcast audio file path."""
        if self.podcast_result and self.podcast_result.script:
            return self.podcast_result.script.audio_path
        return None


@dataclass
class GenerationProgress:
    """Progress tracking for podcast generation."""
    task_id: str
    status: TaskStatus
    current_stage: str
    progress_percentage: float
    estimated_remaining_time: Optional[int] = None  # in seconds
    detailed_status: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)