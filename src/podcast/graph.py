"""Define the Podcast agent graph structure.

This module implements a multi-step podcast generation workflow that:
1. Analyzes the research report content
2. Extracts key points and information
3. Designs appropriate characters for the podcast
4. Generates the podcast script
5. Segments the script for TTS processing
6. Synthesizes speech for each segment
7. Combines audio segments into final podcast
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from loguru import logger

from podcast.context import Context
from podcast.state import InputState, State
from podcast.tools import (
    analyze_research_report,
    combine_audio_segments,
    design_podcast_characters,
    generate_podcast_script,
    save_podcast_script,
    segment_script_for_tts,
    synthesize_speech_qwen,
    _async_file_exists,
)
from shared.models import DialogSegment, PodcastScript, Character
from shared.utils import ensure_directory, generate_timestamped_filename


async def analyze_report_content(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Analyze the input research report to extract key information."""
    try:
        logger.info("Starting report content analysis")
        
        if not state.input_report:
            return {"errors": ["No research report provided"]}
        
        # Analyze the report
        analysis_result = await analyze_research_report(state.input_report)
        
        if "error" in analysis_result:
            return {"errors": [analysis_result["error"]]}
        
        logger.info(f"Successfully analyzed report: {len(analysis_result.get('key_points', []))} key points found")
        
        return {
            "key_points": analysis_result.get("key_points", []),
            "estimated_duration": analysis_result.get("estimated_duration", 1800),
            "messages": [AIMessage(content=f"Analyzed report and extracted {len(analysis_result.get('key_points', []))} key points")]
        }
        
    except Exception as e:
        logger.error(f"Report analysis failed: {e}")
        return {"errors": [f"Report analysis failed: {str(e)}"]}


async def create_characters(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Design characters for the podcast based on the topic and content."""
    try:
        logger.info("Creating podcast characters")
        
        if not state.key_points:
            return {"errors": ["No key points available for character design"]}
        
        # Extract topic from report
        topic = "未知话题"
        if state.input_report:
            lines = state.input_report.split('\n')
            for line in lines:
                if line.startswith('# '):
                    topic = line[2:].strip()
                    topic = topic.replace(' - 深度研究报告', '')
                    break
        
        # Design characters
        # Type casting to ensure compatibility with function signature
        key_points_typed = cast(List[Dict[str, Any]], state.key_points)
        characters = await design_podcast_characters(topic, key_points_typed)
        
        if not characters:
            return {"errors": ["Failed to design podcast characters"]}
        
        logger.info(f"Successfully created {len(characters)} characters")
        
        return {
            "characters": characters,
            "messages": [AIMessage(content=f"Created {len(characters)} characters for the podcast")]
        }
        
    except Exception as e:
        logger.error(f"Character creation failed: {e}")
        return {"errors": [f"Character creation failed: {str(e)}"]}


async def create_script(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate the podcast script from the analyzed content."""
    try:
        logger.info("Generating podcast script")
        
        if not state.key_points or not state.characters:
            return {"errors": ["Missing key points or characters for script generation"]}
        
        # Extract topic
        topic = "未知话题"
        if state.input_report:
            lines = state.input_report.split('\n')
            for line in lines:
                if line.startswith('# '):
                    topic = line[2:].strip()
                    topic = topic.replace(' - 深度研究报告', '')
                    break
        
        # Generate script  
        # Type casting to ensure compatibility with function signatures
        key_points_typed = cast(List[Dict[str, Any]], state.key_points)
        characters_typed = cast(List[Character], state.characters)
        script_content = await generate_podcast_script(
            topic=topic,
            key_points=key_points_typed,
            characters=characters_typed,
            target_duration=state.estimated_duration or runtime.context.default_duration
        )
        
        if not script_content or len(script_content) < 500:
            return {"errors": ["Generated script is too short or empty"]}
        
        # Save script
        script_path = await save_podcast_script(script_content, topic)
        
        logger.info(f"Successfully generated script with {len(script_content)} characters")
        
        return {
            "script_content": script_content,
            "script_path": script_path,
            "messages": [AIMessage(content=f"Generated podcast script with {len(script_content)} characters")]
        }
        
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        return {"errors": [f"Script generation failed: {str(e)}"]}


async def process_audio_segments(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Segment script and process audio for each segment."""
    try:
        logger.info("Processing audio segments")
        
        if not state.script_content:
            return {"errors": ["No script content available for audio processing"]}
        
        # Segment the script
        segments = await segment_script_for_tts(state.script_content)
        
        if not segments:
            return {"errors": ["Failed to segment script for TTS processing"]}
        
        logger.info(f"Script segmented into {len(segments)} parts")
        
        # Process each segment
        audio_segments = []
        failed_segments = 0
        
        # Determine voice models from characters
        host_voice = "xiaoyun"
        guest_voice = "xiaogang"
        
        for character in state.characters:
            role = character.role.lower()
            voice_model = character.voice_config.get("model", "xiaoyun")
            
            if "主持" in role or "host" in role:
                host_voice = voice_model
            elif "嘉宾" in role or "guest" in role or "专家" in role:
                guest_voice = voice_model
        
        logger.info(f"Voice assignment: Host={host_voice}, Guest={guest_voice}")
        
        # Process segments in batches to avoid overwhelming the TTS service
        batch_size = 5
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            
            # Process batch concurrently
            tasks = []
            for segment in batch:
                # Assign voice based on speaker keywords
                speaker_lower = segment.speaker.lower()
                if "主持" in speaker_lower or "host" in speaker_lower:
                    voice_model = host_voice
                elif "嘉宾" in speaker_lower or "专家" in speaker_lower or "guest" in speaker_lower:
                    voice_model = guest_voice
                else:
                    # Alternate between voices for unknown speakers
                    voice_model = host_voice if segment.segment_id % 2 == 1 else guest_voice
                
                output_filename = f"segment_{segment.segment_id:03d}.mp3"
                task = synthesize_speech_qwen(
                    text=segment.content,
                    voice_model=voice_model,
                    output_path=output_filename
                )
                tasks.append((segment, task))
            
            # Wait for batch completion
            for segment, task in tasks:
                try:
                    audio_path = await task
                    if audio_path:
                        segment.audio_file = audio_path
                        audio_segments.append({
                            "segment_id": segment.segment_id,
                            "speaker": segment.speaker,
                            "audio_file": audio_path,
                            "duration": segment.duration_estimate
                        })
                    else:
                        failed_segments += 1
                        logger.warning(f"Failed to synthesize audio for segment {segment.segment_id}")
                except Exception as e:
                    failed_segments += 1
                    logger.error(f"Error processing segment {segment.segment_id}: {e}")
            
            # Small delay between batches
            if i + batch_size < len(segments):
                await asyncio.sleep(1)
        
        success_rate = (len(audio_segments) / len(segments)) * 100 if segments else 0
        logger.info(f"Audio processing completed: {len(audio_segments)}/{len(segments)} segments successful ({success_rate:.1f}%)")
        
        if not audio_segments:
            return {"errors": ["No audio segments were successfully generated"]}
        
        return {
            "audio_segments": audio_segments,
            "messages": [AIMessage(content=f"Processed {len(audio_segments)} audio segments successfully")]
        }
        
    except Exception as e:
        logger.error(f"Audio segment processing failed: {e}")
        return {"errors": [f"Audio segment processing failed: {str(e)}"]}


async def combine_final_audio(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Combine all audio segments into the final podcast audio file."""
    try:
        logger.info("Combining final audio")
        
        if not state.audio_segments:
            return {"errors": ["No audio segments available for combining"]}
        
        # Extract topic for filename
        topic = "podcast"
        if state.input_report:
            lines = state.input_report.split('\n')
            for line in lines:
                if line.startswith('# '):
                    topic = line[2:].strip()
                    topic = topic.replace(' - 深度研究报告', '').replace(' ', '_')
                    break
        
        # Prepare output path
        output_dir = await ensure_directory(Path(runtime.context.output_base_dir) / "final_podcasts")
        audio_filename = generate_timestamped_filename("podcast", topic, ".mp3")
        final_audio_path = output_dir / audio_filename
        
        # Extract audio file paths in order
        audio_files = []
        sorted_segments = sorted(state.audio_segments, key=lambda x: x.get("segment_id", 0))
        
        for segment in sorted_segments:
            audio_file = segment.get("audio_file")
            if audio_file and await _async_file_exists(Path(audio_file)):
                audio_files.append(audio_file)
            else:
                logger.warning(f"Audio file not found for segment {segment.get('segment_id')}")
        
        if not audio_files:
            return {"errors": ["No valid audio files found for combining"]}
        
        # Combine audio files
        success = await combine_audio_segments(audio_files, str(final_audio_path))
        
        if not success:
            return {"errors": ["Failed to combine audio segments"]}
        
        # Calculate total duration
        total_duration = sum(segment.get("duration", 0) for segment in state.audio_segments)
        
        # Cleanup temporary files if configured
        if runtime.context.cleanup_temp:
            try:
                for audio_file in audio_files:
                    await asyncio.to_thread(
                        lambda f=audio_file: Path(f).unlink(missing_ok=True)
                    )
                logger.debug("Cleaned up temporary audio files")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary files: {e}")
        
        logger.info(f"Successfully created final podcast audio: {final_audio_path}")
        
        return {
            "final_audio_path": str(final_audio_path),
            "total_duration": total_duration,
            "messages": [AIMessage(content=f"Created final podcast audio: {final_audio_path.name}")]
        }
        
    except Exception as e:
        logger.error(f"Final audio combination failed: {e}")
        return {"errors": [f"Final audio combination failed: {str(e)}"]}


def should_continue_podcast(state: State) -> Literal["create_characters", "create_script", "process_audio_segments", "combine_final_audio", "__end__"]:
    """Determine the next step in the podcast generation workflow."""
    
    # Check for errors
    if state.errors:
        logger.error(f"Podcast workflow stopped due to errors: {state.errors}")
        return "__end__"
    
    # Check if we have key points but no characters
    if state.key_points and not state.characters:
        return "create_characters"
    
    # Check if we have characters but no script
    if state.characters and not state.script_content:
        return "create_script"
    
    # Check if we have script but no audio segments
    if state.script_content and not state.audio_segments:
        return "process_audio_segments"
    
    # Check if we have audio segments but no final audio
    if state.audio_segments and not state.final_audio_path:
        return "combine_final_audio"
    
    # All steps completed
    return "__end__"


# Create the graph
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Add nodes
builder.add_node("analyze_report_content", analyze_report_content)
builder.add_node("create_characters", create_characters)
builder.add_node("create_script", create_script)
builder.add_node("process_audio_segments", process_audio_segments)
builder.add_node("combine_final_audio", combine_final_audio)

# Set entry point
builder.add_edge("__start__", "analyze_report_content")

# Add conditional edges
builder.add_conditional_edges(
    "analyze_report_content",
    should_continue_podcast,
)

builder.add_conditional_edges(
    "create_characters",
    should_continue_podcast,
)

builder.add_conditional_edges(
    "create_script",
    should_continue_podcast,
)

builder.add_conditional_edges(
    "process_audio_segments",
    should_continue_podcast,
)

builder.add_conditional_edges(
    "combine_final_audio",
    should_continue_podcast,
)

# Compile the graph
graph = builder.compile(name="Podcast Agent")