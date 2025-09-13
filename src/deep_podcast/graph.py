"""Define the Deep Podcast controller graph structure.

This module implements the main controller that coordinates:
1. Deep Research sub-graph for generating research reports
2. Podcast sub-graph for converting reports to podcast content  
3. Progress tracking and error handling
4. Final result compilation and delivery

This implementation uses LangGraph's subgraph functionality to directly
compose the deep_research and podcast graphs as nodes in the main workflow.
"""

import asyncio
import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from loguru import logger

from deep_podcast.context import Context
from deep_podcast.prompts import (
    FINAL_RESULT_PROMPT,
    PROGRESS_UPDATE_PROMPT,
    TOPIC_VALIDATION_PROMPT,
)
from deep_podcast.state import InputState, State
from deep_podcast.utils import load_chat_model
from shared.models import (
    CompletePodcastResult,
    GenerationProgress,
    TaskStatus,
)
from shared.utils import generate_task_id, ensure_directory

# Import subgraphs
from deep_research.graph import graph as research_graph
from podcast.graph import graph as podcast_graph

async def initialize_task(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Initialize the podcast generation task with validation and setup."""
    try:
        logger.info(f"Initializing podcast generation task for topic: {state.user_topic}")
        
        if not state.user_topic or len(state.user_topic.strip()) < 5:
            return {"errors": ["Please provide a valid research topic (at least 5 characters)"]}
        
        # Generate unique task ID
        task_id = generate_task_id()
        
        # Validate topic
        model = load_chat_model()
        validation_prompt = TOPIC_VALIDATION_PROMPT.format(topic=state.user_topic)
        
        validation_response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=validation_prompt)])
        )
        
        # Setup output directory
        output_base_dir = getattr(runtime.context, 'output_base_dir', './outputs')
        output_dir = await ensure_directory(Path(output_base_dir) / f"task_{task_id}")
        
        # Initialize progress tracking
        progress = GenerationProgress(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            current_stage="初始化",
            progress_percentage=10.0,
            detailed_status="任务初始化完成，开始深度研究阶段"
        )
        
        logger.info(f"Task {task_id} initialized successfully")
        
        return {
            "task_id": task_id,
            "progress": progress,
            "output_directory": str(output_dir),
            "start_time": datetime.now(UTC),
            "research_status": TaskStatus.IN_PROGRESS,
            "messages": [validation_response]
        }
        
    except Exception as e:
        logger.error(f"Task initialization failed: {e}")
        return {"errors": [f"Task initialization failed: {str(e)}"]}


async def execute_research_subgraph(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute the deep research subgraph and map its results."""
    try:
        logger.info(f"Starting research subgraph for task {state.task_id}")
        
        # Update progress
        progress = GenerationProgress(
            task_id=state.task_id,
            status=TaskStatus.IN_PROGRESS,
            current_stage="深度研究",
            progress_percentage=25.0,
            detailed_status="正在进行网络搜索和内容分析"
        )
        
        # Import research state and context for proper input
        from deep_research.state import InputState as ResearchInputState
        from deep_research.context import Context as ResearchContext
        
        # Prepare research input
        research_input = ResearchInputState(
            topic=state.user_topic,
            messages=[HumanMessage(content=f"请对'{state.user_topic}'进行深度研究分析")]
        )
        
        # Execute research subgraph with proper context
        research_result = await research_graph.ainvoke(
            research_input,
            config={
                "configurable": {
                    "thread_id": f"research_{state.task_id}",
                }
            },
            context=ResearchContext()
        )
        
        # Check research results
        if research_result.get("errors"):
            logger.error(f"Research subgraph failed: {research_result['errors']}")
            return {
                "research_status": TaskStatus.FAILED,
                "errors": research_result["errors"]
            }
        
        if not research_result.get("final_report"):
            logger.error("Research subgraph completed but no report generated")
            return {
                "research_status": TaskStatus.FAILED,
                "errors": ["Research completed but no report was generated"]
            }
        
        # Update progress
        progress.progress_percentage = 50.0
        progress.current_stage = "研究完成"
        progress.detailed_status = "深度研究完成，准备开始播客生成阶段"
        
        logger.info(f"Research subgraph completed for task {state.task_id}")
        
        return {
            "research_status": TaskStatus.COMPLETED,
            "research_report": research_result["final_report"],
            "research_report_path": research_result.get("report_path", ""),
            "podcast_status": TaskStatus.IN_PROGRESS,
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Research subgraph failed: {e}")
        return {
            "research_status": TaskStatus.FAILED,
            "errors": [f"Research subgraph failed: {str(e)}"]
        }


async def execute_podcast_subgraph(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Execute the podcast generation subgraph and map its results."""
    try:
        logger.info(f"Starting podcast subgraph for task {state.task_id}")
        
        if not state.research_report:
            return {"errors": ["No research report available for podcast generation"]}
        
        # Update progress
        progress = GenerationProgress(
            task_id=state.task_id,
            status=TaskStatus.IN_PROGRESS,
            current_stage="播客生成",
            progress_percentage=60.0,
            detailed_status="正在生成播客脚本和音频"
        )
        
        # Import podcast state and context for proper input
        from podcast.state import InputState as PodcastInputState
        from podcast.context import Context as PodcastContext
        
        # Prepare podcast input
        podcast_input = PodcastInputState(
            input_report=state.research_report,
            messages=[HumanMessage(content=f"请基于研究报告生成播客内容")]
        )
        
        # Execute podcast subgraph with proper context
        podcast_result = await podcast_graph.ainvoke(
            podcast_input,
            config={
                "configurable": {
                    "thread_id": f"podcast_{state.task_id}"
                }
            },
            context=PodcastContext()
        )
        
        # Check podcast results
        if podcast_result.get("errors"):
            logger.error(f"Podcast subgraph failed: {podcast_result['errors']}")
            return {
                "podcast_status": TaskStatus.FAILED,
                "errors": podcast_result["errors"]
            }
        
        # Update progress
        progress.progress_percentage = 90.0
        progress.current_stage = "播客生成完成"
        progress.detailed_status = "播客内容生成完成，正在整理输出"
        
        logger.info(f"Podcast subgraph completed for task {state.task_id}")
        
        return {
            "podcast_status": TaskStatus.COMPLETED,
            "podcast_script": podcast_result.get("script_content", ""),
            "podcast_script_path": podcast_result.get("script_path", ""),
            "podcast_audio_path": podcast_result.get("final_audio_path", ""),
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Podcast subgraph failed: {e}")
        return {
            "podcast_status": TaskStatus.FAILED,
            "errors": [f"Podcast subgraph failed: {str(e)}"]
        }


async def finalize_results(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Finalize the podcast generation results and create the final output."""
    try:
        logger.info(f"Finalizing results for task {state.task_id}")
        
        completion_time = datetime.now(UTC)
        total_time = completion_time - state.start_time if state.start_time else None
        
        # Create final result
        final_result = CompletePodcastResult(
            status=TaskStatus.COMPLETED,
            total_processing_time=total_time.total_seconds() if total_time else None
        )
        
        # Generate completion summary
        model = load_chat_model()
        summary_prompt = FINAL_RESULT_PROMPT.format(
            topic=state.user_topic,
            start_time=state.start_time.strftime("%Y-%m-%d %H:%M:%S") if state.start_time else "Unknown",
            completion_time=completion_time.strftime("%Y-%m-%d %H:%M:%S"),
            total_time=f"{total_time.total_seconds():.1f}秒" if total_time else "Unknown",
            report_path=state.research_report_path,
            script_path=state.podcast_script_path,
            audio_path=state.podcast_audio_path
        )
        
        summary_response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=summary_prompt)])
        )
        
        # Final progress update
        final_progress = GenerationProgress(
            task_id=state.task_id,
            status=TaskStatus.COMPLETED,
            current_stage="完成",
            progress_percentage=100.0,
            detailed_status="播客生成任务已完成"
        )
        
        logger.info(f"Task {state.task_id} completed successfully")
        
        return {
            "final_result": final_result,
            "completion_time": completion_time,
            "progress": final_progress,
            "messages": [summary_response]
        }
        
    except Exception as e:
        logger.error(f"Result finalization failed: {e}")
        return {"errors": [f"Result finalization failed: {str(e)}"]}


def should_continue_workflow(state: State) -> Literal["execute_research_subgraph", "execute_podcast_subgraph", "finalize_results", "__end__"]:
    """Determine the next step in the podcast generation workflow."""
    
    # Check for errors
    if state.errors:
        logger.error(f"Workflow stopped due to errors: {state.errors}")
        return "__end__"
    
    # Check research phase
    if state.research_status == TaskStatus.PENDING or state.research_status == TaskStatus.IN_PROGRESS:
        return "execute_research_subgraph"
    
    if state.research_status == TaskStatus.FAILED:
        return "__end__"
    
    # Check podcast phase
    if (state.research_status == TaskStatus.COMPLETED and 
        (state.podcast_status == TaskStatus.PENDING or state.podcast_status == TaskStatus.IN_PROGRESS)):
        return "execute_podcast_subgraph"
    
    if state.podcast_status == TaskStatus.FAILED:
        return "__end__"
    
    # Both phases completed, finalize
    if (state.research_status == TaskStatus.COMPLETED and 
        state.podcast_status == TaskStatus.COMPLETED and 
        not state.final_result):
        return "finalize_results"
    
    # All done
    return "__end__"


# Create the graph
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Add nodes
builder.add_node("initialize_task", initialize_task)
builder.add_node("execute_research_subgraph", execute_research_subgraph)
builder.add_node("execute_podcast_subgraph", execute_podcast_subgraph)
builder.add_node("finalize_results", finalize_results)

# Set entry point
builder.add_edge("__start__", "initialize_task")

# Add conditional edges
builder.add_conditional_edges(
    "initialize_task",
    should_continue_workflow,
)

builder.add_conditional_edges(
    "execute_research_subgraph",
    should_continue_workflow,
)

builder.add_conditional_edges(
    "execute_podcast_subgraph",
    should_continue_workflow,
)

builder.add_conditional_edges(
    "finalize_results",
    should_continue_workflow,
)

# Compile the graph
graph = builder.compile(name="Deep Podcast Controller")
