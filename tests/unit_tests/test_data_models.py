"""Unit tests for data models."""

import pytest
from datetime import datetime
from shared.models import (
    ResearchReport,
    Character,
    DialogSegment,
    PodcastScript,
    LLMConfig,
    TTSConfig,
    SearchConfig,
    OutputConfig,
    SystemConfig,
    TaskStatus,
    ResearchDepth,
    PodcastStyle,
)


class TestResearchReport:
    """Test ResearchReport data model."""
    
    def test_create_research_report(self):
        """Test creating a research report."""
        report = ResearchReport(
            topic="AI发展趋势",
            summary="AI技术快速发展",
            key_findings=["深度学习突破", "应用场景扩展"],
            sections={"概述": "AI概述内容", "趋势": "发展趋势分析"},
            sources=["https://example.com/ai-news"],
            metadata={"author": "AI研究员"},
            created_at=datetime.now(),
            file_path="/path/to/report.md"
        )
        
        assert report.topic == "AI发展趋势"
        assert len(report.key_findings) == 2
        assert "概述" in report.sections
        assert report.file_path == "/path/to/report.md"
    
    def test_to_dict_conversion(self):
        """Test converting research report to dictionary."""
        report = ResearchReport(
            topic="测试话题",
            summary="测试摘要",
            key_findings=["发现1"],
            sections={"section1": "内容1"},
            sources=["source1"],
            metadata={"test": True},
            created_at=datetime(2024, 1, 1)
        )
        
        report_dict = report.to_dict()
        
        assert report_dict["topic"] == "测试话题"
        assert report_dict["summary"] == "测试摘要"
        assert "created_at" in report_dict
        assert isinstance(report_dict["created_at"], str)
    
    def test_from_dict_conversion(self):
        """Test creating research report from dictionary."""
        data = {
            "topic": "测试话题",
            "summary": "测试摘要",
            "key_findings": ["发现1"],
            "sections": {"section1": "内容1"},
            "sources": ["source1"],
            "metadata": {"test": True},
            "created_at": "2024-01-01T00:00:00",
            "file_path": None
        }
        
        report = ResearchReport.from_dict(data)
        
        assert report.topic == "测试话题"
        assert isinstance(report.created_at, datetime)


class TestCharacter:
    """Test Character data model."""
    
    def test_create_character(self):
        """Test creating a character."""
        character = Character(
            name="主持人小云",
            role="主持人",
            personality="专业、友善",
            voice_config={"model": "xiaoyun", "rate": 1.0},
            background="资深媒体人",
            expertise=["访谈", "主持"]
        )
        
        assert character.name == "主持人小云"
        assert character.role == "主持人"
        assert character.voice_config["model"] == "xiaoyun"
        assert len(character.expertise) == 2
    
    def test_to_dict_conversion(self):
        """Test converting character to dictionary."""
        character = Character(
            name="测试角色",
            role="测试",
            personality="测试性格",
            voice_config={"model": "test"},
            background="测试背景",
            expertise=["技能1", "技能2"]
        )
        
        char_dict = character.to_dict()
        
        assert char_dict["name"] == "测试角色"
        assert char_dict["expertise"] == ["技能1", "技能2"]
        assert char_dict["voice_config"]["model"] == "test"


class TestDialogSegment:
    """Test DialogSegment data model."""
    
    def test_create_dialog_segment(self):
        """Test creating a dialog segment."""
        segment = DialogSegment(
            segment_id=1,
            speaker="主持人",
            content="欢迎来到我们的节目",
            emotion="friendly",
            duration_estimate=5,
            audio_file="/path/to/audio.mp3",
            voice_config={"rate": 1.0}
        )
        
        assert segment.segment_id == 1
        assert segment.speaker == "主持人"
        assert segment.emotion == "friendly"
        assert segment.audio_file == "/path/to/audio.mp3"
    
    def test_to_dict_conversion(self):
        """Test converting dialog segment to dictionary."""
        segment = DialogSegment(
            segment_id=1,
            speaker="测试",
            content="测试内容",
            emotion="neutral",
            duration_estimate=10
        )
        
        seg_dict = segment.to_dict()
        
        assert seg_dict["segment_id"] == 1
        assert seg_dict["speaker"] == "测试"
        assert seg_dict["content"] == "测试内容"


class TestPodcastScript:
    """Test PodcastScript data model."""
    
    def test_create_podcast_script(self):
        """Test creating a podcast script."""
        characters = [
            Character("主持人", "host", "专业", {"model": "voice1"}),
            Character("嘉宾", "guest", "专业", {"model": "voice2"})
        ]
        
        segments = [
            DialogSegment(1, "主持人", "开场白", "friendly", 10),
            DialogSegment(2, "嘉宾", "回应", "neutral", 8)
        ]
        
        script = PodcastScript(
            title="测试播客",
            characters=characters,
            segments=segments,
            total_duration=1800,
            metadata={"topic": "测试话题"},
            script_path="/path/to/script.md",
            audio_path="/path/to/audio.mp3"
        )
        
        assert script.title == "测试播客"
        assert len(script.characters) == 2
        assert len(script.segments) == 2
        assert script.total_duration == 1800
    
    def test_to_dict_conversion(self):
        """Test converting podcast script to dictionary."""
        characters = [Character("主持人", "host", "专业", {"model": "voice1"})]
        segments = [DialogSegment(1, "主持人", "内容", "neutral", 10)]
        
        script = PodcastScript(
            title="测试",
            characters=characters,
            segments=segments,
            total_duration=600,
            metadata={}
        )
        
        script_dict = script.to_dict()
        
        assert script_dict["title"] == "测试"
        assert len(script_dict["characters"]) == 1
        assert len(script_dict["segments"]) == 1
        assert "created_at" in script_dict


class TestConfigurations:
    """Test configuration data models."""
    
    def test_llm_config(self):
        """Test LLM configuration."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet",
            api_key="test_key",
            temperature=0.7,
            max_tokens=4000
        )
        
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet"
        assert config.temperature == 0.7
        assert config.max_tokens == 4000
    
    def test_tts_config(self):
        """Test TTS configuration."""
        config = TTSConfig(
            provider="qwen",
            api_key="test_key",
            voice_models={"host": "xiaoyun", "guest": "xiaogang"},
            speech_rate=1.0
        )
        
        assert config.provider == "qwen"
        assert "host" in config.voice_models
        assert config.speech_rate == 1.0
    
    def test_search_config(self):
        """Test search configuration."""
        config = SearchConfig(
            max_search_results=20,
            max_crawl_pages=50,
            search_timeout=30
        )
        
        assert config.max_search_results == 20
        assert config.max_crawl_pages == 50
        assert config.search_timeout == 30
    
    def test_output_config(self):
        """Test output configuration."""
        config = OutputConfig(
            base_dir="./outputs",
            report_format="markdown",
            audio_format="mp3"
        )
        
        assert config.base_dir == "./outputs"
        assert config.report_format == "markdown"
        assert config.audio_format == "mp3"
        
        # Test get_output_dir method
        output_dir = config.get_output_dir("test")
        assert "test" in str(output_dir)
    
    def test_system_config(self):
        """Test complete system configuration."""
        llm_config = LLMConfig("anthropic", "claude", "key")
        tts_config = TTSConfig()
        search_config = SearchConfig()
        output_config = OutputConfig()
        
        system_config = SystemConfig(
            llm_config=llm_config,
            tts_config=tts_config,
            search_config=search_config,
            output_config=output_config
        )
        
        assert system_config.llm_config.provider == "anthropic"
        assert system_config.tts_config.provider == "qwen"
        assert system_config.search_config.max_search_results == 20
        assert system_config.output_config.base_dir == "./outputs"


class TestEnums:
    """Test enum types."""
    
    def test_task_status(self):
        """Test TaskStatus enum."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
    
    def test_research_depth(self):
        """Test ResearchDepth enum."""
        assert ResearchDepth.BASIC.value == "basic"
        assert ResearchDepth.STANDARD.value == "standard"
        assert ResearchDepth.COMPREHENSIVE.value == "comprehensive"
    
    def test_podcast_style(self):
        """Test PodcastStyle enum."""
        assert PodcastStyle.CONVERSATIONAL.value == "conversational"
        assert PodcastStyle.INTERVIEW.value == "interview"
        assert PodcastStyle.NARRATIVE.value == "narrative"
        assert PodcastStyle.EDUCATIONAL.value == "educational"


if __name__ == "__main__":
    pytest.main([__file__])