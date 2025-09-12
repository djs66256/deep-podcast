"""Pytest configuration and shared fixtures."""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add src directory to Python path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests") 
    config.addinivalue_line("markers", "performance: performance tests")
    config.addinivalue_line("markers", "real_services: tests using real external services")


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory fixture."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_output_dir():
    """Temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response fixture."""
    def _create_response(content: str):
        response = MagicMock()
        response.content = content
        return response
    return _create_response


@pytest.fixture
def sample_research_report():
    """Sample research report for testing."""
    return """# AI技术发展 - 深度研究报告

## 执行摘要

AI技术在过去几年中取得了显著进展，特别是在机器学习和深度学习领域。

## 关键发现

- 深度学习技术日趋成熟
- 自然语言处理能力大幅提升
- AI在各行业的应用不断扩展

## 详细分析

### 技术发展现状

AI技术的发展可以分为几个关键阶段：

1. **基础理论阶段**：建立了机器学习的理论基础
2. **算法突破阶段**：深度学习算法的重大突破
3. **应用普及阶段**：AI技术在各个领域的广泛应用

### 应用领域

AI技术已经在以下领域取得重要应用：

- **自然语言处理**：聊天机器人、机器翻译
- **计算机视觉**：图像识别、自动驾驶
- **推荐系统**：个性化推荐、智能搜索

## 未来展望

随着算力的提升和算法的优化，AI技术将在更多领域发挥重要作用。

## 结论

AI技术的发展前景广阔，将深刻改变人类社会的各个方面。

## 参考来源

1. [AI发展报告](https://example.com/ai-report) - example.com
2. [机器学习趋势](https://example.com/ml-trends) - example.com
3. [深度学习应用](https://example.com/dl-apps) - example.com

---

*本报告基于网络公开信息整理生成，仅供参考。*
"""


@pytest.fixture
def sample_podcast_script():
    """Sample podcast script for testing."""
    return """# AI技术发展 - 播客对话脚本

## 节目信息
- 标题：AI技术发展深度解析
- 时长：30分钟
- 主要话题：AI技术的现状与未来

## 角色介绍
- 主持人：张主持 - 资深科技媒体人
- 嘉宾：李博士 - AI领域专家

## 对话内容

### 开场 (2-3分钟)

[张主持]: 大家好，欢迎收听我们的科技深度解析节目。我是主持人张主持。今天我们邀请到了AI领域的专家李博士，来和我们聊聊AI技术的发展现状。

[李博士]: 大家好，我是李博士，很高兴能够和大家分享AI技术的最新进展。

[张主持]: 李博士，能否先为我们介绍一下当前AI技术的整体发展情况？

### 主体内容

#### 第一部分：AI技术现状

[李博士]: 当前AI技术的发展可以说是日新月异。特别是在深度学习领域，我们看到了很多突破性的进展。

[张主持]: 您能具体谈谈哪些方面的突破最为显著吗？

[李博士]: 我认为最显著的突破主要体现在三个方面：自然语言处理、计算机视觉，以及推荐系统。

#### 第二部分：技术应用

[张主持]: 这些技术突破对普通用户来说意味着什么呢？

[李博士]: 实际上，这些技术已经深入到我们日常生活的方方面面了。比如我们使用的搜索引擎、社交媒体推荐，还有智能助手等等。

### 结尾 (1-2分钟)

[张主持]: 时间过得真快，我们今天的节目就要结束了。最后请李博士为我们总结一下AI技术未来的发展趋势。

[李博士]: 我认为AI技术将继续快速发展，并且会在更多领域发挥重要作用。我们要以开放的心态拥抱这些变化。

[张主持]: 非常感谢李博士的精彩分享，也感谢大家的收听。我们下期节目再见！

---

*本脚本由Deep Podcast系统自动生成，总时长约30分钟*
"""


@pytest.fixture
def mock_system_config():
    """Mock system configuration for tests."""
    from shared.models import SystemConfig, LLMConfig, TTSConfig, SearchConfig, OutputConfig
    
    return SystemConfig(
        llm_config=LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet",
            api_key="test_key",
            temperature=0.7,
            max_tokens=4000
        ),
        tts_config=TTSConfig(
            provider="qwen",
            api_key="test_tts_key",
            voice_models={"host": "xiaoyun", "guest": "xiaogang"}
        ),
        search_config=SearchConfig(
            max_search_results=10,
            max_crawl_pages=20,
            search_timeout=30
        ),
        output_config=OutputConfig(
            base_dir="./test_outputs",
            cleanup_temp=True
        )
    )


@pytest.fixture
def env_setup(monkeypatch):
    """Setup environment variables for testing."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "QWEN_TTS_API_KEY": "test_qwen_key",
        "OUTPUT_BASE_DIR": "./test_outputs",
        "MAX_SEARCH_RESULTS": "10",
        "SEARCH_TIMEOUT": "30",
        "CLEANUP_TEMP": "true"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env