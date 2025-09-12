"""Define the Deep Research agent graph structure.

This module implements a multi-step research workflow that:
1. Analyzes the research topic
2. Generates search queries
3. Performs web searches
4. Crawls and extracts content
5. Analyzes and structures information
6. Generates a comprehensive research report
"""

import json
from datetime import datetime, UTC
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from loguru import logger

from deep_research.context import Context
from deep_research.prompts import (
    CONTENT_ANALYSIS_PROMPT,
    REPORT_GENERATION_PROMPT,
    SEARCH_QUERY_PROMPT,
)
from deep_research.state import InputState, State
from deep_research.tools import (
    batch_crawl_urls,
    duckduckgo_search,
    extract_key_information,
    generate_markdown_report,
    save_research_report,
)
from deep_research.utils import load_chat_model
from shared.utils import ProgressTracker


async def analyze_topic(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze the research topic and generate search queries."""
    try:
        logger.info(f"Analyzing research topic: {state.topic}")
        
        if not state.topic:
            return {"errors": ["No research topic provided"]}
        
        model = load_chat_model(runtime.context.model)
        
        prompt = SEARCH_QUERY_PROMPT.format(topic=state.topic)
        
        response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=prompt)])
        )
        
        # Parse search queries from response
        queries = []
        content = response.content if isinstance(response.content, str) else str(response.content)
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering and clean up
                clean_query = line.lstrip('0123456789.-) ').strip()
                if clean_query and len(clean_query) > 5:
                    queries.append(clean_query)
        
        if not queries:
            # Fallback queries
            queries = [
                f"{state.topic} 概述",
                f"{state.topic} 最新发展",
                f"{state.topic} 应用案例",
                f"{state.topic} 趋势分析",
                f"{state.topic} 专家观点"
            ]
        
        logger.info(f"Generated {len(queries)} search queries")
        
        return {
            "search_queries": queries,
            "messages": [response]
        }
        
    except Exception as e:
        logger.error(f"Topic analysis failed: {e}")
        return {"errors": [f"Topic analysis failed: {str(e)}"]}


async def perform_web_search(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Perform web searches using generated queries."""
    try:
        logger.info("Starting web search phase")
        
        if not state.search_queries:
            return {"errors": ["No search queries available"]}
        
        all_results = []
        
        # Search with each query
        for i, query in enumerate(state.search_queries):
            logger.info(f"Searching query {i+1}/{len(state.search_queries)}: {query}")
            
            results = await duckduckgo_search(
                query, 
                max_results=runtime.context.max_search_results // len(state.search_queries)
            )
            
            all_results.extend(results)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        logger.info(f"Found {len(unique_results)} unique search results")
        
        return {"search_results": unique_results}
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"errors": [f"Web search failed: {str(e)}"]}


async def crawl_content(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Crawl content from search result URLs."""
    try:
        logger.info("Starting content crawling phase")
        
        if not state.search_results:
            return {"errors": ["No search results available for crawling"]}
        
        # Extract URLs from search results
        urls = [result['url'] for result in state.search_results]
        
        # Limit the number of URLs to crawl
        max_crawl = min(len(urls), runtime.context.max_crawl_pages)
        urls = urls[:max_crawl]
        
        logger.info(f"Crawling content from {len(urls)} URLs")
        
        # Crawl content
        crawled_content = await batch_crawl_urls(
            urls, 
            max_concurrent=5  # Use a fixed value since search_config is not available
        )
        
        logger.info(f"Successfully crawled {len(crawled_content)} pages")
        
        return {"crawled_content": crawled_content}
        
    except Exception as e:
        logger.error(f"Content crawling failed: {e}")
        return {"errors": [f"Content crawling failed: {str(e)}"]}


async def analyze_content(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze and structure the crawled content."""
    try:
        logger.info("Starting content analysis phase")
        
        if not state.crawled_content:
            return {"errors": ["No crawled content available for analysis"]}
        
        # Extract key information
        analyzed_content = await extract_key_information(state.crawled_content)
        
        logger.info("Content analysis completed")
        
        return {"analyzed_content": analyzed_content}
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        return {"errors": [f"Content analysis failed: {str(e)}"]}


async def generate_report(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate the final research report."""
    try:
        logger.info("Starting report generation phase")
        
        if not state.analyzed_content:
            return {"errors": ["No analyzed content available for report generation"]}
        
        # Generate report using LLM
        model = load_chat_model(runtime.context.model)
        
        prompt = REPORT_GENERATION_PROMPT.format(
            topic=state.topic,
            analyzed_content=json.dumps(state.analyzed_content, ensure_ascii=False, indent=2)
        )
        
        response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=prompt)])
        )
        
        # Use the LLM-generated report or fallback to template
        content = response.content if isinstance(response.content, str) else str(response.content)
        if content and len(content) > 1000:
            final_report = content
        else:
            # Fallback to template-based report
            sources = []
            for item in state.crawled_content:
                sources.append({
                    'title': item.get('title', 'Unknown Title'),
                    'url': item.get('url', ''),
                    'domain': item.get('domain', 'unknown')
                })
            
            final_report = await generate_markdown_report(
                state.topic,
                state.analyzed_content,
                sources
            )
        
        # Save the report
        report_path = await save_research_report(final_report, state.topic)
        
        logger.info(f"Research report generated and saved to: {report_path}")
        
        return {
            "final_report": final_report,
            "report_path": report_path,
            "messages": [response]
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"errors": [f"Report generation failed: {str(e)}"]}


def should_continue_research(state: State) -> Literal["perform_web_search", "crawl_content", "analyze_content", "generate_report", "__end__"]:
    """Determine the next step in the research workflow."""
    
    # Check for errors
    if state.errors:
        logger.error(f"Research workflow stopped due to errors: {state.errors}")
        return "__end__"
    
    # Check if we have search queries but no results
    if state.search_queries and not state.search_results:
        return "perform_web_search"
    
    # Check if we have search results but no crawled content
    if state.search_results and not state.crawled_content:
        return "crawl_content"
    
    # Check if we have crawled content but no analysis
    if state.crawled_content and not state.analyzed_content:
        return "analyze_content"
    
    # Check if we have analysis but no final report
    if state.analyzed_content and not state.final_report:
        return "generate_report"
    
    # All steps completed
    return "__end__"


# Create the graph
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Add nodes
builder.add_node("analyze_topic", analyze_topic)
builder.add_node("perform_web_search", perform_web_search)
builder.add_node("crawl_content", crawl_content)
builder.add_node("analyze_content", analyze_content)
builder.add_node("generate_report", generate_report)

# Set entry point
builder.add_edge("__start__", "analyze_topic")

# Add conditional edges
builder.add_conditional_edges(
    "analyze_topic",
    should_continue_research,
)

builder.add_conditional_edges(
    "perform_web_search",
    should_continue_research,
)

builder.add_conditional_edges(
    "crawl_content",
    should_continue_research,
)

builder.add_conditional_edges(
    "analyze_content",
    should_continue_research,
)

builder.add_conditional_edges(
    "generate_report",
    should_continue_research,
)

# Compile the graph
graph = builder.compile(name="Deep Research Agent")