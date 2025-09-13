"""Tools for the Podcast agent.

This module provides tools for analyzing research reports, generating podcast scripts,
designing characters, and integrating with TTS services.
"""

from ast import mod
import asyncio
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime, get_runtime
from loguru import logger
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_exponential

# Try to import aiofiles for async file operations
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False


async def _async_file_exists(file_path: Path) -> bool:
    """Async helper to check if file exists."""
    try:
        return await asyncio.to_thread(lambda: file_path.exists())
    except Exception:
        return False


async def _async_write_binary_file(file_path: Path, content: bytes) -> None:
    """Async helper to write binary data to file."""
    if HAS_AIOFILES:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
    else:
        # Fallback to thread-based async operation
        def _write_file():
            with open(file_path, "wb") as f:
                f.write(content)
        await asyncio.to_thread(_write_file)

from podcast.context import Context
from podcast.prompts import (
    AUDIO_SEGMENTATION_PROMPT,
    CHARACTER_DESIGN_PROMPT,
    KEY_POINTS_EXTRACTION_PROMPT,
    SCRIPT_GENERATION_PROMPT,
)
from podcast.utils import (
    clean_dialog_text,
    estimate_reading_time,
    extract_topic_from_report,
    format_duration,
    generate_audio_filename,
    generate_script_filename,
    load_chat_model,
    parse_script_dialog,
    split_long_text,
)
from shared.models import Character, DialogSegment, PodcastScript
from shared.utils import (
    ensure_directory,
    generate_timestamped_filename,
    save_json_file,
    save_text_file,
)


async def analyze_research_report(report_content: str) -> Dict[str, Any]:
    """Analyze research report and extract key points for podcast creation.
    
    Args:
        report_content: The markdown content of the research report
        
    Returns:
        Dictionary containing extracted key points and metadata
    """
    try:
        runtime = get_runtime(Context)
        logger.info("Analyzing research report for podcast content")
        
        if not report_content or len(report_content) < 100:
            return {"error": "Report content is too short or empty"}
        
        model = load_chat_model()
        
        prompt = KEY_POINTS_EXTRACTION_PROMPT.format(report_content=report_content)
        
        response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=prompt)])
        )
        
        # Try to parse as JSON first, fallback to text parsing
        try:
            content = str(response.content)
            # Extract JSON from response if wrapped in other text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                key_points_data = json.loads(json_str)
            else:
                key_points_data = json.loads(content)
        except json.JSONDecodeError:
            # Parse manually from text response
            logger.warning("Failed to parse JSON response, using text parsing fallback")
            content = str(response.content)
            key_points_data = {"key_points": []}
            
            # Try to extract structured information
            sections = content.split('\n\n')
            current_point = None
            
            for section in sections:
                lines = section.strip().split('\n')
                if not lines or not lines[0].strip():
                    continue
                    
                # Look for numbered points or titles
                first_line = lines[0].strip()
                if (re.match(r'^\d+[.)]', first_line) or 
                    '**' in first_line or 
                    first_line.startswith('#') or
                    any(keyword in first_line.lower() for keyword in ['核心', '关键', '重要', '主要', '发现', '观点', '趋势'])):
                    
                    # Extract title
                    title = re.sub(r'^\d+[.)]', '', first_line)
                    title = title.strip('*#').strip()
                    
                    # Get description from remaining lines
                    description_lines = [line.strip() for line in lines[1:] if line.strip()]
                    description = ' '.join(description_lines) if description_lines else title
                    
                    # Estimate importance based on keywords
                    importance = 3  # default
                    if any(word in section.lower() for word in ['核心', '关键', '重要']):
                        importance = 4
                    if any(word in section.lower() for word in ['数据', '发现', '结论']):
                        importance = max(importance, 4)
                        
                    current_point = {
                        "title": title[:100],  # Limit title length
                        "description": description[:500],  # Limit description length
                        "importance": importance,
                        "discussion_angle": f"从{title}的角度讨论"
                    }
                    key_points_data["key_points"].append(current_point)
            
            # If still no key points found, extract from major sections
            if not key_points_data["key_points"]:
                logger.warning("No structured points found, extracting from sections")
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if (line.startswith('##') and '## ' in line):
                        title = line.replace('##', '').strip()
                        if len(title) > 5 and title not in ['背景', '概述', '结论', '参考']:
                            key_points_data["key_points"].append({
                                "title": title,
                                "description": f"关于{title}的详细分析和讨论",
                                "importance": 3,
                                "discussion_angle": f"深入探讨{title}的内容和意义"
                            })
        
        # Extract topic from report
        topic = extract_topic_from_report(report_content)
        
        # Calculate estimated duration
        word_count = len(report_content.split())
        estimated_duration = max(900, min(2700, word_count * 3))  # 15-45 minutes based on content
        
        analysis_result = {
            "topic": topic,
            "key_points": key_points_data.get("key_points", []),
            "word_count": word_count,
            "estimated_duration": estimated_duration,
            "content_sections": _extract_content_sections(report_content),
            "analysis_timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"Successfully analyzed report: {len(analysis_result['key_points'])} key points extracted")
        
        # Debug: log key points for troubleshooting
        if len(analysis_result['key_points']) == 0:
            logger.warning("No key points extracted from report. Content preview:")
            logger.warning(f"Report length: {len(report_content)} characters")
            logger.warning(f"LLM response preview: {str(response.content)[:500]}...")
        else:
            logger.debug(f"Key points titles: {[point.get('title', 'No title') for point in analysis_result['key_points']]}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Report analysis failed: {e}")
        return {"error": f"Report analysis failed: {str(e)}"}


def _extract_content_sections(report_content: str) -> List[Dict[str, str]]:
    """Extract sections from markdown report."""
    sections = []
    lines = report_content.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        if line.startswith('## '):
            if current_section:
                sections.append({
                    "title": current_section,
                    "content": '\n'.join(current_content).strip()
                })
            current_section = line[3:].strip()
            current_content = []
        elif current_section:
            current_content.append(line)
    
    if current_section:
        sections.append({
            "title": current_section,
            "content": '\n'.join(current_content).strip()
        })
    
    return sections


async def design_podcast_characters(topic: str, key_points: List[Dict[str, Any]]) -> List[Character]:
    """Design characters for the podcast based on topic and content.
    
    Args:
        topic: Main topic of the podcast
        key_points: Key points extracted from the research
        
    Returns:
        List of Character objects
    """
    try:
        runtime = get_runtime(Context)
        logger.info(f"Designing podcast characters for topic: {topic}")
        
        model = load_chat_model()
        
        # Analyze topic characteristics
        topic_characteristics = _analyze_topic_characteristics(topic, key_points)
        
        prompt = CHARACTER_DESIGN_PROMPT.format(
            topic=topic,
            topic_characteristics=topic_characteristics
        )
        
        response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=prompt)])
        )
        
        # Parse character data from response
        try:
            content = str(response.content)
            characters_data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback to default characters
            characters_data = _get_default_characters_data(topic)
        
        characters = []
        voice_models = runtime.context.voice_models
        
        # Create Character objects with proper voice assignment
        for i, char_data in enumerate(characters_data.get("characters", [])):
            # Assign voices based on role, not just index
            role = char_data.get("role", "参与者").lower()
            if "主持" in role or "host" in role:
                voice_model = voice_models.get("host", "xiaoyun")
            elif "嘉宾" in role or "guest" in role or "专家" in role:
                voice_model = voice_models.get("guest", "xiaogang")
            else:
                # Fallback to index-based assignment
                voice_model = list(voice_models.values())[i % len(voice_models)]
            
            character = Character(
                name=char_data.get("name", f"角色{i+1}"),
                role=char_data.get("role", "参与者"),
                personality=char_data.get("personality", "专业、友善"),
                background=char_data.get("background", ""),
                expertise=char_data.get("expertise", []),
                voice_config={
                    "model": voice_model,
                    "rate": runtime.context.speech_rate,
                    "pitch": 0,
                    "volume": 1.0
                }
            )
            characters.append(character)
            logger.info(f"Assigned voice '{voice_model}' to {character.name} (role: {character.role})")
        
        logger.info(f"Successfully designed {len(characters)} characters")
        return characters
        
    except Exception as e:
        logger.error(f"Character design failed: {e}")
        # Return default characters as fallback
        runtime = get_runtime(Context)
        return _get_default_characters(topic, runtime.context.voice_models)


def _analyze_topic_characteristics(topic: str, key_points: List[Dict[str, Any]]) -> str:
    """Analyze topic characteristics for character design."""
    characteristics = []
    
    # Topic type analysis
    if any(word in topic.lower() for word in ['技术', '科技', 'ai', '人工智能', '区块链']):
        characteristics.append("技术导向")
    if any(word in topic.lower() for word in ['商业', '市场', '经济', '投资']):
        characteristics.append("商业导向")
    if any(word in topic.lower() for word in ['教育', '学习', '培训']):
        characteristics.append("教育导向")
    
    # Complexity analysis
    complexity_score = sum(len(point.get("description", "").split()) for point in key_points)
    if complexity_score > 500:
        characteristics.append("内容复杂")
    else:
        characteristics.append("内容适中")
    
    return ", ".join(characteristics) if characteristics else "通用话题"


def _get_default_characters_data(topic: str) -> Dict[str, Any]:
    """Get default character data structure."""
    return {
        "characters": [
            {
                "name": "主持人小云",
                "role": "主持人",
                "personality": "专业、引导性强、善于提问和总结",
                "background": "资深媒体人，擅长深度访谈",
                "expertise": ["访谈技巧", "内容梳理", "观众互动"]
            },
            {
                "name": "专家小刚",
                "role": "专业嘉宾",
                "personality": "知识渊博、表达清晰、有独特见解",
                "background": f"{topic}领域专家，具有丰富的实践经验",
                "expertise": [topic, "行业分析", "趋势预测"]
            }
        ]
    }


def _get_default_characters(topic: str, voice_models: Optional[Dict[str, str]] = None) -> List[Character]:
    """Get default character configuration."""
    if voice_models is None:
        voice_models = {"host": "xiaoyun", "guest": "xiaogang"}
    
    return [
        Character(
            name="主持人小云",
            role="主持人",
            personality="专业、引导性强、善于提问和总结",
            background="资深媒体人，擅长深度访谈",
            expertise=["访谈技巧", "内容梳理", "观众互动"],
            voice_config={
                "model": voice_models.get("host", "xiaoyun"),
                "rate": 1.0,
                "pitch": 0,
                "volume": 1.0
            }
        ),
        Character(
            name="专家小刚",
            role="嘉宾",
            personality="知识渊博、表达清晰、有独特见解",
            background=f"{topic}领域专家，具有丰富的实践经验",
            expertise=[topic, "行业分析", "趋势预测"],
            voice_config={
                "model": voice_models.get("guest", "xiaogang"),
                "rate": 1.0,
                "pitch": 0,
                "volume": 1.0
            }
        )
    ]


async def generate_podcast_script(
    topic: str, 
    key_points: List[Dict[str, Any]], 
    characters: List[Character],
    target_duration: int = 1800
) -> str:
    """Generate podcast script from analyzed content.
    
    Args:
        topic: Main topic of the podcast
        key_points: Key points to cover
        characters: Character definitions
        target_duration: Target duration in seconds
        
    Returns:
        Generated script in markdown format
    """
    try:
        runtime = get_runtime(Context)
        logger.info(f"Generating podcast script for topic: {topic}")
        
        model = load_chat_model()
        
        prompt = SCRIPT_GENERATION_PROMPT.format(
            topic=topic,
            key_points=json.dumps(key_points, ensure_ascii=False, indent=2),
            characters=json.dumps([char.to_dict() for char in characters], ensure_ascii=False, indent=2),
            target_duration=target_duration // 60  # Convert to minutes
        )
        
        response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=prompt)])
        )
        
        script_content = str(response.content)
        
        # Validate and enhance script if needed
        if len(script_content) < 1000:
            script_content = _generate_fallback_script(topic, key_points, characters, target_duration)
        
        logger.info(f"Successfully generated script with {len(script_content)} characters")
        return script_content
        
    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        return _generate_fallback_script(topic, key_points, characters, target_duration)


def _generate_fallback_script(
    topic: str, 
    key_points: List[Dict[str, Any]], 
    characters: List[Character],
    target_duration: int
) -> str:
    """Generate a fallback script when LLM generation fails."""
    host = characters[0] if characters else Character(name="主持人", role="主持人", personality="专业", voice_config={})
    guest = characters[1] if len(characters) > 1 else Character(name="嘉宾", role="嘉宾", personality="专业", voice_config={})
    
    timestamp = datetime.now().strftime("%Y年%m月%d日")
    
    script = f"""# {topic} - 播客对话脚本

## 节目信息
- 标题：深度解析{topic}
- 时长：{target_duration // 60}分钟
- 录制日期：{timestamp}
- 主要话题：{topic}的深度分析与探讨

## 角色介绍
- 主持人：{host.name} - {host.background or '专业主持人'}
- 嘉宾：{guest.name} - {guest.background or f'{topic}领域专家'}

## 对话内容

### 开场 (2-3分钟)

[{host.name}]: 大家好，欢迎来到我们的深度解析节目。我是{host.name}。今天我们要和大家探讨一个非常有意思的话题——{topic}。

[{guest.name}]: 大家好，我是{guest.name}，很高兴能和大家分享关于{topic}的一些见解和思考。

[{host.name}]: 那么首先，能否请{guest.name}为我们简单介绍一下{topic}的基本概念？

### 主体内容

"""
    
    # Add content sections based on key points
    for i, point in enumerate(key_points[:5], 1):
        title = point.get("title", f"要点{i}")
        description = point.get("description", "")
        
        script += f"""#### 第{i}部分：{title}

[{guest.name}]: {description[:200]}...

[{host.name}]: 这确实很有意思。那么您认为这对我们普通人来说意味着什么呢？

[{guest.name}]: 我觉得这个问题问得很好。从实际应用的角度来看...

"""
    
    script += f"""### 结尾 (2-3分钟)

[{host.name}]: 今天和{guest.name}的对话让我收获很多。最后，能否请您为我们的听众总结一下今天讨论的核心要点？

[{guest.name}]: 当然。今天我们主要讨论了{topic}的几个关键方面...

[{host.name}]: 非常感谢{guest.name}的精彩分享，也感谢大家的收听。我们下期节目再见！

---

*本脚本由Deep Podcast系统自动生成，总时长约{target_duration // 60}分钟*
"""
    
    return script


async def segment_script_for_tts(script_content: str) -> List[DialogSegment]:
    """Segment script into TTS-ready dialog parts.
    
    Args:
        script_content: The podcast script content
        
    Returns:
        List of DialogSegment objects
    """
    try:
        logger.info("Segmenting script for TTS processing")
        
        model = load_chat_model()
        
        prompt = AUDIO_SEGMENTATION_PROMPT.format(script_content=script_content)
        
        response = cast(
            AIMessage,
            await model.ainvoke([HumanMessage(content=prompt)])
        )
        
        # Parse segmentation response
        try:
            content = str(response.content).strip()
            
            segments_data = None
            
            # Try multiple JSON extraction strategies
            # Strategy 1: Direct JSON parsing
            try:
                segments_data = json.loads(content)
            except json.JSONDecodeError:
                # Strategy 2: Extract JSON array with regex
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug(f"Extracted JSON: {json_str[:200]}...")
                    try:
                        segments_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Strategy 3: Fix common JSON issues
                        logger.debug("Trying JSON repair")
                        clean_json = json_str
                        # Fix missing commas between adjacent string properties
                        clean_json = re.sub(r'"(\s*\n\s*)"', '",\\1"', clean_json)
                        # Fix missing commas before closing braces
                        clean_json = re.sub(r'"(\s*\n\s*)}', '",\\1}', clean_json)
                        segments_data = json.loads(clean_json)
                        
            if segments_data is None:
                raise json.JSONDecodeError("All JSON parsing strategies failed", content, 0)
                
        except json.JSONDecodeError as e:
            # Fallback to manual parsing
            # logger.warning(f"Failed to parse JSON response: {str(e)}. Using manual parsing fallback")
            logger.debug(f"Problematic content: {content[:500]}...")
            return _parse_script_manually(script_content)
        
        segments = []
        segments_list = segments_data if isinstance(segments_data, list) else segments_data.get("segments", [])
        
        if not segments_list:
            logger.warning("No segments found in LLM response, falling back to manual parsing")
            return _parse_script_manually(script_content)
        
        for i, segment_data in enumerate(segments_list):
            try:
                segment = DialogSegment(
                    segment_id=segment_data.get("segment_id", i + 1),
                    speaker=segment_data.get("speaker", "Unknown"),
                    content=clean_dialog_text(segment_data.get("content", "")),
                    emotion=segment_data.get("emotion", "neutral"),
                    duration_estimate=segment_data.get("duration_estimate", 10),
                    voice_config=segment_data.get("voice_config", {
                        "rate": 1.0,
                        "pitch": 0,
                        "volume": 1.0
                    })
                )
                
                # Only add segments with actual content
                if segment.content.strip():
                    segments.append(segment)
                    
            except Exception as e:
                logger.warning(f"Failed to create segment {i}: {e}")
                continue
        
        if not segments:
            logger.warning("No valid segments created from LLM response, falling back to manual parsing")
            return _parse_script_manually(script_content)
        
        logger.info(f"Successfully segmented script into {len(segments)} parts")
        return segments
        
    except Exception as e:
        logger.error(f"Script segmentation failed: {str(e)}")
        logger.debug(f"Exception type: {type(e).__name__}")
        # Always fall back to manual parsing
        try:
            return _parse_script_manually(script_content)
        except Exception as fallback_error:
            logger.error(f"Manual parsing also failed: {str(fallback_error)}")
            # Re-raise the exception instead of creating a default segment
            raise fallback_error


def _parse_script_manually(script_content: str) -> List[DialogSegment]:
    """Manually parse script when LLM segmentation fails."""
    segments = []
    
    try:
        logger.debug(f"Manual parsing script content preview: {script_content[:300]}...")
        dialog_parts = parse_script_dialog(script_content)
        logger.debug(f"Found {len(dialog_parts)} dialog parts")
        
        if dialog_parts:
            logger.debug(f"First dialog part keys: {list(dialog_parts[0].keys()) if dialog_parts else 'N/A'}")
        
        if not dialog_parts:
            logger.error("No dialog parts found in script, manual parsing failed")
            raise ValueError("Script does not contain recognizable dialog structure")
        
        for i, part in enumerate(dialog_parts):
            try:
                # Ensure part is a dictionary with required keys
                if not isinstance(part, dict):
                    logger.warning(f"Invalid dialog part type at index {i}: {type(part)}")
                    continue
                    
                if 'content' not in part or 'speaker' not in part:
                    logger.warning(f"Missing required keys in dialog part {i}: {list(part.keys())}")
                    continue
                    
                content = part["content"]
                speaker = part["speaker"]
                
                # Clean speaker name
                speaker = speaker.strip()
                if not speaker:
                    speaker = f"角色{i+1}"
                
                # Split long content
                content_chunks = split_long_text(content, max_length=200)
                
                for j, chunk in enumerate(content_chunks):
                    if chunk.strip():  # Only create segments for non-empty chunks
                        segment = DialogSegment(
                            segment_id=i * 100 + j + 1,
                            speaker=speaker,
                            content=clean_dialog_text(chunk),
                            emotion="neutral",
                            duration_estimate=estimate_reading_time(chunk),
                            voice_config={
                                "rate": 1.0,
                                "pitch": 0,
                                "volume": 1.0
                            }
                        )
                        segments.append(segment)
                        
            except Exception as part_error:
                logger.warning(f"Failed to process dialog part {i}: {part_error}")
                continue
        
        logger.info(f"Manual parsing created {len(segments)} segments")
        return segments
        
    except Exception as e:
        logger.error(f"Manual script parsing failed: {e}")
        # Re-raise the exception to indicate parsing failure
        raise e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def synthesize_speech_qwen(
    text: str, 
    voice_model: str = "xiaoyun",
    output_path: Optional[str] = None
) -> Optional[str]:
    """Synthesize speech using Qwen TTS service.
    
    Args:
        text: Text content to synthesize
        voice_model: Voice model to use
        output_path: Optional output file path
        
    Returns:
        Path to generated audio file or None if failed
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS synthesis")
            return None
            
        runtime = get_runtime(Context)
        
        if not runtime.context.qwen_tts_api_key:
            logger.error("Qwen TTS API key not configured")
            return None
        
        # Validate TTS configuration
        if not runtime.context.qwen_tts_base_url:
            logger.error("Qwen TTS base URL not configured")
            return None
            
        logger.debug(f"TTS Configuration:")
        logger.debug(f"  API Key: {'***' if runtime.context.qwen_tts_api_key else 'NOT SET'}")
        logger.debug(f"  Base URL: {runtime.context.qwen_tts_base_url}")
        logger.debug(f"  Voice model: {voice_model}")
        logger.debug(f"  Audio format: {runtime.context.audio_format}")
        logger.debug(f"  Speech rate: {runtime.context.speech_rate}")
        logger.debug(f"  Text length: {len(text)} characters")
        
        # Use official API endpoint according to documentation
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        # Prepare request headers according to official documentation
        headers = {
            "Authorization": f"Bearer {runtime.context.qwen_tts_api_key}",
            "Content-Type": "application/json",
        }
        
        # Map voice models to official supported voices
        voice_mapping = {
            "xiaoyun": "Cherry",
            "xiaogang": "Ethan", 
            "cherry": "Cherry",
            "serena": "Serena",
            "ethan": "Ethan",
            "chelsie": "Chelsie",
            "dylan": "Dylan",
            "jada": "Jada",
            "sunny": "Sunny"
        }
        
        # Get mapped voice or use default
        mapped_voice = voice_mapping.get(voice_model.lower(), "Cherry")
        logger.debug(f"  Mapped voice: {voice_model} -> {mapped_voice}")
        
        # Official API request format
        data = {
            "model": "qwen-tts",
            "input": {
                "text": text,
                "voice": mapped_voice
            }
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            
            # Parse response according to official format
            result = response.json()
            logger.debug(f"TTS API response: {result}")
            
            if "output" in result and "audio" in result["output"]:
                audio_info = result["output"]["audio"]
                
                if "url" in audio_info:
                    # Download audio from URL
                    audio_url = audio_info["url"]
                    logger.debug(f"Downloading audio from: {audio_url}")
                    
                    # Create output path if not provided
                    if not output_path:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"temp_audio_{timestamp}.{runtime.context.audio_format}"
                    
                    audio_dir = await ensure_directory(Path(runtime.context.output_base_dir) / "audio_segments")
                    full_path = audio_dir / output_path
                    
                    # Download audio file
                    audio_response = await client.get(audio_url)
                    audio_response.raise_for_status()
                    
                    # Save audio file using async file operations
                    await _async_write_binary_file(full_path, audio_response.content)
                    
                    logger.debug(f"Speech synthesized and saved to: {full_path}")
                    
                    # Log usage information if available
                    if "usage" in result:
                        usage = result["usage"]
                        logger.debug(f"TTS Usage - Total tokens: {usage.get('total_tokens', 'N/A')}, Audio tokens: {usage.get('output_tokens', 'N/A')}")
                    
                    return str(full_path)
                    
                elif "data" in audio_info:
                    # Handle base64 encoded audio data (streaming)
                    import base64
                    audio_data = base64.b64decode(audio_info["data"])
                    
                    # Create output path if not provided
                    if not output_path:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"temp_audio_{timestamp}.{runtime.context.audio_format}"
                    
                    audio_dir = await ensure_directory(Path(runtime.context.output_base_dir) / "audio_segments")
                    full_path = audio_dir / output_path
                    
                    # Save base64 audio data using async file operations
                    await _async_write_binary_file(full_path, audio_data)
                    
                    logger.debug(f"Speech synthesized from base64 data and saved to: {full_path}")
                    return str(full_path)
                else:
                    logger.error("No audio URL or data found in response")
                    raise Exception("Invalid response: no audio content")
            else:
                logger.error(f"Invalid response format: {result}")
                raise Exception(f"Invalid response format: missing output.audio")
            
    except httpx.HTTPStatusError as e:
        error_details = {
            "status_code": e.response.status_code,
            "response_text": e.response.text,
            "url": str(e.request.url),
            "method": e.request.method
        }
        logger.error(f"TTS API HTTP error: {error_details}")
        
        # Parse error response if possible
        try:
            error_json = e.response.json()
            if "message" in error_json:
                logger.error(f"API Error Message: {error_json['message']}")
                if "url error" in error_json["message"].lower():
                    logger.error("This appears to be a URL configuration issue. Please check:")
                    logger.error(f"1. Base URL: {runtime.context.qwen_tts_base_url}")
                    logger.error(f"2. API Key format: {'Valid' if runtime.context.qwen_tts_api_key.startswith('sk-') else 'Check format'}")
                    logger.error("3. Ensure the API endpoint supports the model and parameters being used")
        except:
            logger.error("Could not parse error response")
            
    except httpx.ConnectError as e:
        logger.error(f"TTS API connection error: {e}")
        logger.error(f"Failed to connect to: {runtime.context.qwen_tts_base_url}")
    except httpx.TimeoutException as e:
        logger.error(f"TTS API timeout: {e}")
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        
        # Always try to create a fallback audio file for development/testing
        try:
            # Create output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"temp_audio_{timestamp}.{runtime.context.audio_format}"
            
            # Create a silent audio segment as fallback
            audio_dir = await ensure_directory(Path(runtime.context.output_base_dir) / "audio_segments")
            full_path = audio_dir / output_path
            
            # Calculate reasonable duration based on text
            chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
            if chinese_chars > len(text) * 0.5:
                # Chinese text: ~250 chars/min
                duration_seconds = max(3, chinese_chars / 250 * 60)
            else:
                # English text: ~150 words/min
                words = len(text.split())
                duration_seconds = max(3, words / 150 * 60)
            
            # Create silent audio using async operations
            duration_ms = int(duration_seconds * 1000)
            
            # Use asyncio.to_thread for audio processing to avoid blocking
            await asyncio.to_thread(
                lambda: AudioSegment.silent(duration=duration_ms).export(str(full_path), format="mp3")
            )
            
            logger.warning(f"Created fallback silent audio: {full_path} ({duration_seconds:.1f}s)")
            logger.info(f"Text content: {text[:100]}...")
            return str(full_path)
            
        except Exception as fallback_error:
            logger.error(f"Fallback audio creation also failed: {fallback_error}")
    
    return None


async def combine_audio_segments(audio_files: List[str], output_path: str) -> bool:
    """Combine multiple audio segments into a single file.
    
    Args:
        audio_files: List of audio file paths to combine
        output_path: Output path for combined audio
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Combining {len(audio_files)} audio segments")
        
        # Load and combine audio segments using async operations
        combined = AudioSegment.empty()
        
        for audio_file in audio_files:
            if await _async_file_exists(Path(audio_file)):
                # Load audio segment in thread to avoid blocking
                segment = await asyncio.to_thread(
                    AudioSegment.from_file, audio_file
                )
                combined += segment
                # Add small pause between segments
                combined += AudioSegment.silent(duration=500)  # 0.5 second pause
            else:
                logger.warning(f"Audio file not found: {audio_file}")
        
        if len(combined) > 0:
            # Export combined audio using async operation
            await ensure_directory(Path(output_path).parent)
            await asyncio.to_thread(
                combined.export, output_path, format="mp3"
            )
            
            logger.info(f"Successfully combined audio to: {output_path}")
            return True
        else:
            logger.error("No valid audio segments to combine")
            return False
            
    except Exception as e:
        logger.error(f"Audio combination failed: {e}")
        return False


async def save_podcast_script(script_content: str, topic: str) -> str:
    """Save podcast script to file.
    
    Args:
        script_content: Script content in markdown format
        topic: Topic for filename generation
        
    Returns:
        Path to saved script file
    """
    try:
        runtime = get_runtime(Context)
        output_dir = Path(runtime.context.output_base_dir) / "podcast_scripts"
        
        filename = generate_script_filename(topic)
        file_path = output_dir / filename
        
        success = await save_text_file(script_content, file_path)
        
        if success:
            logger.info(f"Podcast script saved to: {file_path}")
            return str(file_path)
        else:
            logger.error("Failed to save podcast script")
            return ""
            
    except Exception as e:
        logger.error(f"Error saving podcast script: {e}")
        return ""


# Tool function exports for LangGraph
TOOLS: List[Callable[..., Any]] = [
    analyze_research_report,
    design_podcast_characters,
    generate_podcast_script,
    segment_script_for_tts,
    synthesize_speech_qwen,
    combine_audio_segments,
    save_podcast_script,
]