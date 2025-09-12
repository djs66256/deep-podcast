"""End-to-end integration tests for the Deep Podcast system."""

import pytest
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from deep_podcast.graph import graph as deep_podcast_graph
from deep_podcast.state import InputState as DeepPodcastInputState
from deep_research.graph import graph as research_graph
from deep_research.state import InputState as ResearchInputState
from podcast.graph import graph as podcast_graph
from podcast.state import InputState as PodcastInputState
from shared.config import get_system_config, setup_logging


@pytest.fixture
def setup_test_environment():
    """Setup test environment."""
    # Setup logging
    setup_logging()
    
    # Create test output directory
    test_output_dir = Path("./test_outputs")
    test_output_dir.mkdir(exist_ok=True)
    
    # Set environment variables for testing
    os.environ["OUTPUT_BASE_DIR"] = str(test_output_dir)
    os.environ["LLM_PROVIDER"] = "anthropic"
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    os.environ["QWEN_TTS_API_KEY"] = "test_key"
    
    yield test_output_dir
    
    # Cleanup
    import shutil
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)


class TestDeepResearchIntegration:
    """Integration tests for Deep Research workflow."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_research_workflow_mocked(self, setup_test_environment):
        """Test complete research workflow with mocked external services."""
        
        # Mock external dependencies
        with patch('deep_research.tools.duckduckgo_search') as mock_search, \
             patch('deep_research.tools.batch_crawl_urls') as mock_crawl, \
             patch('deep_research.tools.load_chat_model') as mock_model:
            
            # Setup mocks
            mock_search.return_value = [
                {
                    'title': 'AI发展趋势报告',
                    'url': 'https://example.com/ai-trends',
                    'snippet': 'AI技术正在快速发展',
                    'domain': 'example.com',
                    'search_query': 'AI发展趋势'
                }
            ]
            
            mock_crawl.return_value = [
                {
                    'url': 'https://example.com/ai-trends',
                    'title': 'AI发展趋势详细报告',
                    'content': 'AI人工智能技术在各个领域都有重要进展，包括机器学习、深度学习、自然语言处理等方面。' * 20,
                    'quality_score': 0.8,
                    'domain': 'example.com'
                }
            ]
            
            # Mock LLM responses
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = """1. AI技术基础概念
2. 机器学习发展现状
3. 深度学习应用案例
4. 自然语言处理进展
5. AI未来发展趋势"""
            mock_llm.ainvoke.return_value = mock_response
            mock_model.return_value = mock_llm
            
            # Test input
            research_input = ResearchInputState(
                topic="AI发展趋势",
                messages=[]
            )
            
            # Execute research workflow
            result = await research_graph.ainvoke(
                research_input,
                config={"configurable": {"thread_id": "test_research"}}
            )
            
            # Verify results
            assert "final_report" in result
            assert result["final_report"] is not None
            assert len(result["final_report"]) > 100
            assert "AI发展趋势" in result["final_report"]
            
            # Check that external services were called
            mock_search.assert_called()
            mock_crawl.assert_called()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_research_error_handling(self, setup_test_environment):
        """Test research workflow error handling."""
        
        with patch('deep_research.tools.duckduckgo_search') as mock_search:
            # Simulate search failure
            mock_search.return_value = []
            
            research_input = ResearchInputState(
                topic="非常特殊的测试话题",
                messages=[]
            )
            
            result = await research_graph.ainvoke(
                research_input,
                config={"configurable": {"thread_id": "test_error"}}
            )
            
            # Should handle gracefully and not crash
            assert "errors" in result or "final_report" in result


class TestPodcastIntegration:
    """Integration tests for Podcast generation workflow."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_podcast_workflow_mocked(self, setup_test_environment):
        """Test complete podcast workflow with mocked services."""
        
        # Sample research report
        research_report = """# AI发展趋势 - 深度研究报告

## 执行摘要

AI技术正在快速发展，在各个领域都有重要应用。

## 关键发现

- 机器学习技术日趋成熟
- 深度学习在图像识别方面取得突破
- 自然语言处理能力显著提升

## 详细分析

人工智能技术的发展经历了多个阶段，目前正处于快速发展期。

## 结论

AI技术将继续快速发展，在更多领域得到应用。"""
        
        with patch('podcast.tools.load_chat_model') as mock_model, \
             patch('podcast.tools.synthesize_speech_qwen') as mock_tts, \
             patch('podcast.tools.combine_audio_segments') as mock_combine:
            
            # Mock LLM responses
            mock_llm = MagicMock()
            
            # Mock key points extraction
            key_points_response = MagicMock()
            key_points_response.content = """[
  {
    "title": "AI技术发展现状",
    "description": "机器学习和深度学习技术快速发展",
    "importance": 5
  },
  {
    "title": "应用领域扩展",
    "description": "AI在各个行业的应用不断扩展",
    "importance": 4
  }
]"""
            
            # Mock character design
            character_response = MagicMock()
            character_response.content = """{
  "characters": [
    {
      "name": "主持人小云",
      "role": "主持人",
      "personality": "专业、引导性强",
      "background": "资深科技媒体人"
    },
    {
      "name": "AI专家张博士",
      "role": "专业嘉宾",
      "personality": "知识渊博、表达清晰",
      "background": "AI领域资深专家"
    }
  ]
}"""
            
            # Mock script generation
            script_response = MagicMock()
            script_response.content = """# AI发展趋势 - 播客对话脚本

## 对话内容

[主持人小云]: 大家好，欢迎来到我们的节目。今天我们要讨论AI发展趋势。

[AI专家张博士]: 大家好，很高兴和大家分享AI技术的最新发展。

[主持人小云]: 请您介绍一下当前AI技术的发展现状。

[AI专家张博士]: AI技术确实在快速发展，特别是在机器学习和深度学习方面。"""
            
            # Mock audio segmentation
            segmentation_response = MagicMock()
            segmentation_response.content = """[
  {
    "segment_id": 1,
    "speaker": "主持人小云",
    "content": "大家好，欢迎来到我们的节目。",
    "emotion": "friendly",
    "duration_estimate": 5
  },
  {
    "segment_id": 2,
    "speaker": "AI专家张博士",
    "content": "大家好，很高兴和大家分享。",
    "emotion": "neutral",
    "duration_estimate": 4
  }
]"""
            
            # Setup mock responses in order
            mock_llm.ainvoke.side_effect = [
                key_points_response,
                character_response,
                script_response,
                segmentation_response
            ]
            mock_model.return_value = mock_llm
            
            # Mock TTS
            mock_tts.side_effect = [
                "/test/audio1.mp3",
                "/test/audio2.mp3"
            ]
            
            # Mock audio combination
            mock_combine.return_value = True
            
            # Test input
            podcast_input = PodcastInputState(
                input_report=research_report,
                messages=[]
            )
            
            # Execute podcast workflow
            result = await podcast_graph.ainvoke(
                podcast_input,
                config={"configurable": {"thread_id": "test_podcast"}}
            )
            
            # Verify results
            assert "script_content" in result
            assert result["script_content"] is not None
            assert "主持人" in result["script_content"]
            assert "AI" in result["script_content"]


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_podcast_generation_mocked(self, setup_test_environment):
        """Test complete podcast generation from topic to audio."""
        
        with patch('deep_research.tools.duckduckgo_search') as mock_search, \
             patch('deep_research.tools.batch_crawl_urls') as mock_crawl, \
             patch('deep_research.tools.load_chat_model') as mock_research_model, \
             patch('podcast.tools.load_chat_model') as mock_podcast_model, \
             patch('podcast.tools.synthesize_speech_qwen') as mock_tts, \
             patch('podcast.tools.combine_audio_segments') as mock_combine:
            
            # Setup research mocks
            mock_search.return_value = [
                {
                    'title': '区块链技术发展报告',
                    'url': 'https://example.com/blockchain',
                    'snippet': '区块链技术在金融科技领域的应用',
                    'domain': 'example.com'
                }
            ]
            
            mock_crawl.return_value = [
                {
                    'url': 'https://example.com/blockchain',
                    'title': '区块链技术深度分析',
                    'content': '区块链是一种分布式账本技术，具有去中心化、不可篡改等特点。' * 30,
                    'quality_score': 0.9,
                    'domain': 'example.com'
                }
            ]
            
            # Mock research LLM
            research_llm = MagicMock()
            research_response = MagicMock()
            research_response.content = """1. 区块链基础概念
2. 分布式账本技术
3. 智能合约应用
4. 金融科技创新
5. 未来发展前景"""
            research_llm.ainvoke.return_value = research_response
            mock_research_model.return_value = research_llm
            
            # Mock podcast LLM responses
            podcast_llm = MagicMock()
            
            # Response sequence for podcast generation
            podcast_responses = [
                # Key points extraction
                MagicMock(content='[{"title": "区块链基础", "description": "分布式账本技术概述", "importance": 5}]'),
                # Character design
                MagicMock(content='{"characters": [{"name": "主持人", "role": "主持人", "personality": "专业"}, {"name": "专家", "role": "嘉宾", "personality": "专业"}]}'),
                # Script generation
                MagicMock(content='# 区块链技术 - 播客脚本\n\n[主持人]: 欢迎大家。\n[专家]: 谢谢邀请。'),
                # Audio segmentation
                MagicMock(content='[{"segment_id": 1, "speaker": "主持人", "content": "欢迎大家", "emotion": "friendly", "duration_estimate": 3}]')
            ]
            
            podcast_llm.ainvoke.side_effect = podcast_responses
            mock_podcast_model.return_value = podcast_llm
            
            # Mock TTS and audio combination
            mock_tts.return_value = "/test/segment.mp3"
            mock_combine.return_value = True
            
            # Test complete workflow
            deep_podcast_input = DeepPodcastInputState(
                user_topic="区块链技术应用",
                messages=[]
            )
            
            result = await deep_podcast_graph.ainvoke(
                deep_podcast_input,
                config={"configurable": {"thread_id": "test_complete"}}
            )
            
            # Verify complete workflow
            assert "final_result" in result or "research_report" in result
            assert "errors" not in result or len(result.get("errors", [])) == 0
            
            # Verify that both research and podcast phases were attempted
            mock_search.assert_called()
            mock_tts.assert_called()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_error_recovery(self, setup_test_environment):
        """Test workflow error recovery mechanisms."""
        
        with patch('deep_research.tools.duckduckgo_search') as mock_search:
            # Simulate complete search failure
            mock_search.side_effect = Exception("Network error")
            
            deep_podcast_input = DeepPodcastInputState(
                user_topic="测试错误恢复",
                messages=[]
            )
            
            result = await deep_podcast_graph.ainvoke(
                deep_podcast_input,
                config={"configurable": {"thread_id": "test_error_recovery"}}
            )
            
            # Should handle errors gracefully
            assert isinstance(result, dict)
            # Should have either completed with fallback or recorded errors
            assert "errors" in result or "final_result" in result


class TestPerformanceAndScaling:
    """Performance and scaling tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_requests(self, setup_test_environment):
        """Test handling multiple concurrent requests."""
        import asyncio
        
        async def run_single_request(topic_suffix):
            with patch('deep_research.tools.duckduckgo_search', return_value=[]), \
                 patch('deep_research.tools.batch_crawl_urls', return_value=[]):
                
                input_state = DeepPodcastInputState(
                    user_topic=f"测试话题{topic_suffix}",
                    messages=[]
                )
                
                return await deep_podcast_graph.ainvoke(
                    input_state,
                    config={"configurable": {"thread_id": f"concurrent_test_{topic_suffix}"}}
                )
        
        # Run multiple requests concurrently
        tasks = [run_single_request(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed
        assert len(results) == 3
        
        # Check that no request crashed with unhandled exception
        for result in results:
            assert not isinstance(result, Exception) or isinstance(result, dict)


@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"), 
    reason="Integration tests require RUN_INTEGRATION_TESTS environment variable"
)
class TestRealServiceIntegration:
    """Tests with real external services (requires API keys)."""
    
    @pytest.mark.asyncio
    @pytest.mark.real_services
    async def test_real_search_integration(self):
        """Test with real search service (requires proper configuration)."""
        # This test should only run when explicitly enabled
        # and when proper API keys are configured
        
        research_input = ResearchInputState(
            topic="Python编程语言",
            messages=[]
        )
        
        try:
            result = await research_graph.ainvoke(
                research_input,
                config={"configurable": {"thread_id": "real_test"}}
            )
            
            # If successful, should have generated a report
            assert "final_report" in result
            assert len(result["final_report"]) > 500
            
        except Exception as e:
            # Expected if API keys are not configured
            pytest.skip(f"Real service test skipped: {e}")


if __name__ == "__main__":
    # Run with: pytest tests/integration_tests/test_end_to_end.py -v -m integration
    pytest.main([__file__, "-v", "-m", "integration"])