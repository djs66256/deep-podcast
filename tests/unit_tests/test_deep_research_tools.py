"""Unit tests for deep research tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from deep_research.tools import (
    duckduckgo_search,
    fetch_webpage_content,
    batch_crawl_urls,
    extract_key_information,
    generate_markdown_report,
    save_research_report,
)


class TestDuckDuckGoSearch:
    """Test DuckDuckGo search functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Test successful search operation."""
        with patch('deep_research.tools.DDGS') as mock_ddgs:
            # Mock search results
            mock_results = [
                {
                    'title': 'Test Article 1',
                    'href': 'https://example.com/1',
                    'body': 'Test content 1'
                },
                {
                    'title': 'Test Article 2', 
                    'href': 'https://example.com/2',
                    'body': 'Test content 2'
                }
            ]
            
            mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_results
            
            # Mock runtime context
            with patch('deep_research.tools.get_runtime') as mock_runtime:
                mock_runtime.return_value.context.max_search_results = 10
                
                results = await duckduckgo_search("test query", max_results=5)
                
                assert len(results) == 2
                assert results[0]['title'] == 'Test Article 1'
                assert results[0]['url'] == 'https://example.com/1'
                assert 'domain' in results[0]
    
    @pytest.mark.asyncio
    async def test_search_with_invalid_urls(self):
        """Test search filtering invalid URLs."""
        with patch('deep_research.tools.DDGS') as mock_ddgs:
            mock_results = [
                {
                    'title': 'Valid Article',
                    'href': 'https://example.com/valid',
                    'body': 'Valid content'
                },
                {
                    'title': 'Invalid Article',
                    'href': 'invalid-url',
                    'body': 'Invalid content'
                }
            ]
            
            mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_results
            
            with patch('deep_research.tools.get_runtime') as mock_runtime:
                mock_runtime.return_value.context.max_search_results = 10
                
                results = await duckduckgo_search("test query")
                
                # Only valid URL should be included
                assert len(results) == 1
                assert results[0]['url'] == 'https://example.com/valid'
    
    @pytest.mark.asyncio
    async def test_search_exception_handling(self):
        """Test search exception handling."""
        with patch('deep_research.tools.DDGS') as mock_ddgs:
            mock_ddgs.side_effect = Exception("Search failed")
            
            results = await duckduckgo_search("test query")
            
            assert results == []


class TestWebpageContentFetch:
    """Test webpage content fetching."""
    
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful webpage fetch."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <main>
                    <h1>Main Content</h1>
                    <p>This is the main content of the page.</p>
                </main>
            </body>
        </html>
        """
        
        with patch('deep_research.tools.httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.text = html_content
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.raise_for_status = MagicMock()
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await fetch_webpage_content("https://example.com")
            
            assert result is not None
            assert result['title'] == 'Test Page'
            assert 'Main Content' in result['content']
            assert result['url'] == 'https://example.com'
            assert 'quality_score' in result
    
    @pytest.mark.asyncio
    async def test_fetch_non_html_content(self):
        """Test fetching non-HTML content."""
        with patch('deep_research.tools.httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.headers = {'content-type': 'application/json'}
            mock_response.raise_for_status = MagicMock()
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await fetch_webpage_content("https://example.com/api")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        """Test fetch timeout handling."""
        with patch('deep_research.tools.httpx.AsyncClient') as mock_client:
            from httpx import TimeoutException
            mock_client.return_value.__aenter__.return_value.get.side_effect = TimeoutException("Timeout")
            
            result = await fetch_webpage_content("https://example.com")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_http_error(self):
        """Test HTTP error handling."""
        with patch('deep_research.tools.httpx.AsyncClient') as mock_client:
            from httpx import HTTPStatusError
            mock_response = MagicMock()
            mock_response.status_code = 404
            
            mock_client.return_value.__aenter__.return_value.get.side_effect = HTTPStatusError(
                "Not found", request=MagicMock(), response=mock_response
            )
            
            result = await fetch_webpage_content("https://example.com/404")
            
            assert result is None


class TestBatchCrawlUrls:
    """Test batch URL crawling."""
    
    @pytest.mark.asyncio
    async def test_successful_batch_crawl(self):
        """Test successful batch crawling."""
        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3"
        ]
        
        # Mock successful content fetch
        async def mock_fetch(url, timeout):
            return {
                'url': url,
                'title': f'Title for {url}',
                'content': 'Test content that is long enough to pass minimum length filter.',
                'quality_score': 0.8
            }
        
        with patch('deep_research.tools.fetch_webpage_content', side_effect=mock_fetch):
            with patch('deep_research.tools.get_runtime') as mock_runtime:
                mock_runtime.return_value.context.crawl_timeout = 15
                mock_runtime.return_value.context.content_min_length = 10
                
                results = await batch_crawl_urls(urls, max_concurrent=2)
                
                assert len(results) == 3
                assert all('url' in result for result in results)
                assert all('content' in result for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_crawl_with_failures(self):
        """Test batch crawling with some failures."""
        urls = [
            "https://example.com/success",
            "https://example.com/fail",
            "https://example.com/short"
        ]
        
        async def mock_fetch(url, timeout):
            if 'fail' in url:
                return None
            elif 'short' in url:
                return {
                    'url': url,
                    'content': 'Short',  # Too short
                    'quality_score': 0.5
                }
            else:
                return {
                    'url': url,
                    'content': 'Long enough content to pass the minimum length filter.',
                    'quality_score': 0.8
                }
        
        with patch('deep_research.tools.fetch_webpage_content', side_effect=mock_fetch):
            with patch('deep_research.tools.get_runtime') as mock_runtime:
                mock_runtime.return_value.context.crawl_timeout = 15
                mock_runtime.return_value.context.content_min_length = 20
                
                results = await batch_crawl_urls(urls)
                
                # Only successful and long enough content should be returned
                assert len(results) == 1
                assert 'success' in results[0]['url']


class TestKeyInformationExtraction:
    """Test key information extraction."""
    
    @pytest.mark.asyncio
    async def test_extract_from_content_list(self):
        """Test extracting information from content list."""
        content_list = [
            {
                'url': 'https://example.com/1',
                'title': 'AI Development Trends',
                'content': 'Artificial intelligence is developing rapidly with machine learning and deep learning technologies.',
                'domain': 'example.com',
                'quality_score': 0.8
            },
            {
                'url': 'https://example.com/2',
                'title': 'Future of AI',
                'content': 'The future of artificial intelligence includes automation and intelligent systems.',
                'domain': 'example.com',
                'quality_score': 0.7
            }
        ]
        
        result = await extract_key_information(content_list)
        
        assert 'total_sources' in result
        assert 'total_words' in result
        assert 'average_quality' in result
        assert 'key_terms' in result
        assert 'domains_covered' in result
        assert result['total_sources'] == 2
        assert len(result['domains_covered']) == 1
    
    @pytest.mark.asyncio
    async def test_extract_from_empty_list(self):
        """Test extracting from empty content list."""
        result = await extract_key_information([])
        
        assert 'summary' in result
        assert result['summary'] == "No content available for analysis"


class TestMarkdownReportGeneration:
    """Test markdown report generation."""
    
    @pytest.mark.asyncio
    async def test_generate_report(self):
        """Test generating markdown report."""
        topic = "人工智能发展趋势"
        analyzed_content = {
            'total_sources': 3,
            'total_words': 1500,
            'average_quality': 0.75,
            'domains_covered': ['example.com', 'ai-news.com'],
            'key_terms': [
                {'term': 'artificial', 'frequency': 10},
                {'term': 'intelligence', 'frequency': 8}
            ],
            'summary': 'AI技术快速发展'
        }
        sources = [
            {'title': 'AI News 1', 'url': 'https://example.com/1', 'domain': 'example.com'},
            {'title': 'AI News 2', 'url': 'https://example.com/2', 'domain': 'example.com'}
        ]
        
        report = await generate_markdown_report(topic, analyzed_content, sources)
        
        assert topic in report
        assert '# ' in report  # Header present
        assert '## ' in report  # Sections present
        assert str(analyzed_content['total_sources']) in report
        assert 'https://example.com/1' in report
        assert '深度研究报告' in report
    
    @pytest.mark.asyncio
    async def test_generate_report_with_minimal_data(self):
        """Test generating report with minimal data."""
        topic = "测试话题"
        analyzed_content = {}
        sources = []
        
        report = await generate_markdown_report(topic, analyzed_content, sources)
        
        assert topic in report
        assert '# ' in report
        assert len(report) > 100  # Should have basic structure


class TestSaveResearchReport:
    """Test saving research reports."""
    
    @pytest.mark.asyncio
    async def test_successful_save(self):
        """Test successful report saving."""
        report_content = "# Test Report\n\nThis is a test report."
        topic = "测试话题"
        
        with patch('deep_research.tools.get_runtime') as mock_runtime:
            mock_runtime.return_value.context.output_base_dir = "./test_outputs"
            
            with patch('deep_research.tools.save_text_file', return_value=True) as mock_save:
                result = await save_research_report(report_content, topic)
                
                assert result != ""
                assert "research" in result
                assert ".md" in result
                mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_failed_save(self):
        """Test failed report saving."""
        report_content = "# Test Report"
        topic = "测试话题"
        
        with patch('deep_research.tools.get_runtime') as mock_runtime:
            mock_runtime.return_value.context.output_base_dir = "./test_outputs"
            
            with patch('deep_research.tools.save_text_file', return_value=False):
                result = await save_research_report(report_content, topic)
                
                assert result == ""


if __name__ == "__main__":
    pytest.main([__file__])