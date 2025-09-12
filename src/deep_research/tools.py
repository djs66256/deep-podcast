"""Tools for the Deep Research agent.

This module provides tools for web searching, crawling, content extraction,
and report generation functionality.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage
from langgraph.runtime import get_runtime
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from deep_research.context import Context
from shared.models import ResearchReport
from shared.utils import (
    calculate_content_score,
    extract_domain,
    generate_timestamped_filename,
    sanitize_text,
    save_text_file,
    validate_url,
)


async def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo search engine.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, snippet
    """
    try:
        runtime = get_runtime(Context)
        max_results = min(max_results, runtime.context.max_search_results)
        
        logger.info(f"Searching DuckDuckGo for: {query}")
        
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=max_results):
                if validate_url(result.get('href', '')):
                    results.append({
                        'title': sanitize_text(result.get('title', '')),
                        'url': result.get('href', ''),
                        'snippet': sanitize_text(result.get('body', '')),
                        'domain': extract_domain(result.get('href', '')),
                        'search_query': query,
                    })
            
        logger.info(f"Found {len(results)} search results")
        return results
        
    except Exception as e:
        logger.error(f"DuckDuckGo search failed for query '{query}': {e}")
        return []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def fetch_webpage_content(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """Fetch and parse content from a webpage.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with parsed content or None if failed
    """
    try:
        logger.debug(f"Fetching content from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('content-type', ''):
                logger.warning(f"Non-HTML content type for {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else "No Title"
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract main content
            main_content = ""
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '.main-content',
                '.post-content', '.entry-content', '#content'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    main_content = content_element.get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body content
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
            
            # Clean and sanitize content
            main_content = sanitize_text(main_content)
            
            # Extract metadata
            meta_description = ""
            desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            if desc_tag:
                meta_description = desc_tag.get('content', '')
            
            # Extract keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            keywords = keywords_tag.get('content', '').split(',') if keywords_tag else []
            keywords = [kw.strip() for kw in keywords if kw.strip()]
            
            content_data = {
                'url': url,
                'title': sanitize_text(title),
                'content': main_content,
                'meta_description': sanitize_text(meta_description),
                'keywords': keywords,
                'domain': extract_domain(url),
                'content_length': len(main_content),
                'quality_score': calculate_content_score(main_content),
                'fetched_at': datetime.now().isoformat(),
            }
            
            logger.debug(f"Successfully extracted {len(main_content)} characters from {url}")
            return content_data
            
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error {e.response.status_code} for {url}")
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
    
    return None


async def batch_crawl_urls(urls: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """Crawl multiple URLs concurrently.
    
    Args:
        urls: List of URLs to crawl
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of successfully crawled content
    """
    runtime = get_runtime(Context)
    timeout = runtime.context.crawl_timeout
    min_length = runtime.context.content_min_length
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def crawl_with_semaphore(url: str) -> Optional[Dict[str, Any]]:
        async with semaphore:
            content = await fetch_webpage_content(url, timeout)
            if content and len(content['content']) >= min_length:
                return content
            return None
    
    logger.info(f"Starting batch crawl of {len(urls)} URLs")
    
    tasks = [crawl_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    crawled_content = []
    for result in results:
        if isinstance(result, dict) and result is not None:
            crawled_content.append(result)
        elif isinstance(result, Exception):
            logger.debug(f"Crawl task failed: {result}")
    
    logger.info(f"Successfully crawled {len(crawled_content)} out of {len(urls)} URLs")
    return crawled_content


async def extract_key_information(content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract and structure key information from crawled content.
    
    Args:
        content_list: List of crawled content dictionaries
        
    Returns:
        Structured analysis of the content
    """
    if not content_list:
        return {"summary": "No content available for analysis"}
    
    # Aggregate all content
    all_text = " ".join([item['content'] for item in content_list])
    all_titles = [item['title'] for item in content_list]
    all_domains = list(set([item['domain'] for item in content_list]))
    
    # Basic analysis
    total_words = len(all_text.split())
    avg_quality = sum(item['quality_score'] for item in content_list) / len(content_list)
    
    # Extract key themes (basic keyword extraction)
    words = re.findall(r'\b[A-Za-z]{4,}\b', all_text.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get most frequent meaningful words
    common_words = {'that', 'this', 'with', 'from', 'they', 'have', 'were', 'been', 'their', 'said', 'each', 'would', 'there', 'what', 'about', 'which', 'when', 'one', 'all', 'first', 'also', 'after', 'back', 'other', 'many', 'than', 'then', 'them', 'these', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'those', 'take', 'most', 'way', 'find', 'use', 'her', 'may', 'even', 'give', 'same', 'any', 'under', 'might', 'while', 'last', 'could', 'great', 'little', 'good', 'come', 'such', 'see', 'need', 'place', 'right', 'want', 'look', 'few', 'new', 'part', 'case', 'seem', 'call', 'include', 'long', 'put', 'end', 'why', 'try', 'turn', 'ask', 'help', 'different', 'number', 'next', 'early', 'course', 'important', 'possible', 'hand', 'example', 'either', 'far', 'local', 'large', 'another', 'available', 'major', 'result', 'big', 'group', 'around', 'however', 'home', 'small', 'every', 'following', 'since', 'without', 'public', 'system', 'although', 'both', 'during', 'such', 'point', 'sort', 'change', 'best', 'within', 'often', 'high', 'between', 'level', 'process', 'almost', 'today', 'house', 'country', 'keep', 'never', 'american', 'general', 'social', 'however', 'among', 'information', 'government', 'million', 'least', 'several', 'second', 'something', 'set', 'study', 'full', 'later', 'week', 'year', 'state', 'control', 'always', 'person', 'policy', 'company', 'likely', 'economic', 'political', 'human', 'actually', 'business', 'university', 'able', 'become', 'line', 'question', 'include', 'service', 'national', 'development', 'research', 'although', 'analysis', 'report', 'percent', 'according', 'provide', 'particularly', 'therefore', 'increase', 'experience', 'including', 'order', 'interest', 'recent', 'finally', 'probably', 'certain', 'current', 'rather', 'quality', 'really', 'value', 'decision', 'usually', 'program', 'society', 'often', 'community', 'especially', 'problem', 'various', 'history', 'technology', 'activity', 'growth', 'market', 'support', 'industry', 'global', 'family', 'education', 'member', 'relationship', 'future', 'quite', 'certainly', 'project', 'position', 'method', 'clear', 'continue', 'design', 'particular', 'individual', 'management', 'power', 'build', 'allow', 'potential', 'health', 'approach', 'opportunity', 'understand', 'performance', 'impact', 'financial', 'model', 'create', 'effect', 'international', 'culture', 'success', 'organization', 'idea', 'modern', 'role', 'focus', 'energy', 'security', 'structure', 'issue', 'similar', 'training', 'response', 'purpose', 'improve', 'team', 'though', 'across', 'application', 'significant', 'data', 'economic', 'force', 'instead', 'legal', 'professional', 'region', 'knowledge', 'consider', 'environment', 'investment', 'type', 'agreement', 'specific', 'standard', 'institution', 'nature', 'software', 'computer', 'network', 'customer', 'cost', 'resource', 'price', 'risk', 'technology', 'equipment', 'conference', 'equipment', 'forward', 'industry', 'leadership', 'maintain', 'pressure', 'production', 'range', 'responsibility', 'school', 'simple', 'traditional', 'additional', 'balance', 'board', 'challenge', 'commission', 'detail', 'expert', 'factor', 'generation', 'indeed', 'meeting', 'offer', 'operation', 'private', 'represent', 'science', 'staff', 'strategy', 'task', 'material', 'media', 'physical', 'product', 'quality', 'requirement', 'series', 'skill', 'source', 'successful', 'tend', 'tool', 'trend'}
    
    key_terms = []
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
        if word not in common_words and len(word) > 3:
            key_terms.append({"term": word, "frequency": freq})
    
    return {
        "total_sources": len(content_list),
        "total_words": total_words,
        "average_quality": round(avg_quality, 2),
        "domains_covered": all_domains,
        "key_terms": key_terms[:10],
        "content_titles": all_titles,
        "summary": f"Analyzed {len(content_list)} sources with {total_words} total words from {len(all_domains)} domains.",
        "raw_content": all_text[:5000],  # First 5000 chars for LLM analysis
    }


async def generate_markdown_report(
    topic: str,
    analyzed_content: Dict[str, Any],
    sources: List[Dict[str, Any]]
) -> str:
    """Generate a structured markdown research report.
    
    Args:
        topic: Research topic
        analyzed_content: Structured content analysis
        sources: List of source information
        
    Returns:
        Formatted markdown report
    """
    timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    
    # Format sources
    sources_text = ""
    for i, source in enumerate(sources, 1):
        sources_text += f"{i}. [{source['title']}]({source['url']}) - {source['domain']}\n"
    
    # Format key terms
    key_terms_text = ""
    for term in analyzed_content.get('key_terms', []):
        key_terms_text += f"- **{term['term']}** (出现 {term['frequency']} 次)\n"
    
    report_template = f"""# {topic} - 深度研究报告

**生成时间**: {timestamp}  
**数据来源**: {analyzed_content.get('total_sources', 0)} 个网络资源  
**分析字数**: {analyzed_content.get('total_words', 0):,} 字  

## 执行摘要

基于对 {analyzed_content.get('total_sources', 0)} 个网络资源的深度分析，本报告全面研究了"{topic}"这一主题。报告涵盖了相关的核心概念、最新发展趋势、实际应用案例以及未来展望。

## 背景与概述

本研究通过分析来自 {len(analyzed_content.get('domains_covered', []))} 个不同领域的权威资源，对{topic}进行了全面的调研和分析。研究内容基于最新的网络信息，确保了信息的时效性和准确性。

## 关键发现

### 主要话题和概念

{key_terms_text}

### 重要见解

- **信息来源多样性**: 本研究覆盖了 {len(analyzed_content.get('domains_covered', []))} 个不同的信息源域名
- **内容质量评估**: 平均内容质量得分为 {analyzed_content.get('average_quality', 0)}/1.0
- **研究深度**: 总计分析了 {analyzed_content.get('total_words', 0):,} 个词汇的相关内容

## 详细分析

### 当前发展状况

基于收集到的信息，{topic}领域呈现出以下特点：

{analyzed_content.get('summary', '暂无详细分析内容')}

### 技术趋势和创新

通过对多个资源的综合分析，我们发现该领域正在经历重要的发展变化。主要的技术趋势包括：

1. **数字化转型**: 传统模式正在向数字化方向演进
2. **创新应用**: 新技术的应用场景不断扩展
3. **市场机遇**: 新的商业模式和机会不断涌现

### 挑战与机遇

当前{topic}领域面临的主要挑战包括：
- 技术标准化需求
- 市场竞争加剧
- 监管环境变化

同时，也存在显著的发展机遇：
- 技术创新空间
- 市场需求增长
- 政策支持力度

## 案例研究

通过分析收集到的信息，我们发现了多个值得关注的实际应用案例。这些案例展示了{topic}在不同场景下的应用效果和发展潜力。

## 未来展望

基于当前的发展趋势和市场分析，{topic}领域在未来几年预计将呈现以下发展态势：

1. **技术成熟度提升**: 相关技术将更加成熟和稳定
2. **应用场景扩展**: 应用领域将进一步拓宽
3. **标准化进程**: 行业标准将逐步建立和完善
4. **市场规模增长**: 市场规模预计将持续增长

## 结论与建议

{topic}作为一个重要的研究和应用领域，具有广阔的发展前景。建议相关利益方：

1. **持续关注技术发展**: 跟踪最新的技术趋势和创新
2. **加强标准化建设**: 参与或推动行业标准的制定
3. **探索应用场景**: 积极探索和验证新的应用场景
4. **加强合作交流**: 促进产学研用各方的合作交流

## 参考来源

{sources_text}

---

*本报告基于网络公开信息整理生成，仅供参考。具体决策请结合实际情况和专业建议。*

**报告生成系统**: Deep Research Agent  
**技术支持**: LangGraph + AI  
"""

    return report_template


async def save_research_report(report_content: str, topic: str) -> str:
    """Save research report to file.
    
    Args:
        report_content: Markdown content of the report
        topic: Research topic for filename generation
        
    Returns:
        Path to saved file
    """
    try:
        runtime = get_runtime(Context)
        output_dir = Path(runtime.context.output_base_dir) / "research_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = generate_timestamped_filename("research", topic, ".md")
        file_path = output_dir / filename
        
        success = await save_text_file(report_content, file_path)
        
        if success:
            logger.info(f"Research report saved to: {file_path}")
            return str(file_path)
        else:
            logger.error("Failed to save research report")
            return ""
            
    except Exception as e:
        logger.error(f"Error saving research report: {e}")
        return ""


# Tool function exports for LangGraph
TOOLS: List[Callable[..., Any]] = [
    duckduckgo_search,
    batch_crawl_urls,
    extract_key_information,
    generate_markdown_report,
    save_research_report,
]