# API 参考文档

## 概述

深度播客生成系统提供了三个主要的智能体API，以及相关的工具和配置接口。本文档详细描述了各个组件的API接口。

## 主要组件

### Deep Podcast Controller (主控制器)

主控制器协调整个播客生成流程，从研究话题到最终音频输出。

#### 输入状态 (InputState)

```python
@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    user_topic: str = ""  # 用户提供的研究话题
```

#### 完整状态 (State)

```python
@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
    
    # 任务管理
    task_id: str = ""
    progress: Optional[GenerationProgress] = None
    
    # 研究阶段
    research_status: TaskStatus = TaskStatus.PENDING
    research_report: str = ""
    research_report_path: str = ""
    
    # 播客生成阶段  
    podcast_status: TaskStatus = TaskStatus.PENDING
    podcast_script: str = ""
    podcast_script_path: str = ""
    podcast_audio_path: str = ""
    
    # 输出管理
    output_directory: str = ""
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # 错误处理
    error_message: str = ""
    errors: list[str] = field(default_factory=list)
    
    # 最终结果
    final_result: Optional[CompletePodcastResult] = None
```

#### 使用示例

```python
from deep_podcast.graph import graph
from deep_podcast.state import InputState

# 创建输入状态
input_state = InputState(
    user_topic="人工智能在医疗领域的应用",
    messages=[]
)

# 执行完整流程
result = await graph.ainvoke(
    input_state,
    config={"configurable": {"thread_id": "medical_ai_podcast"}}
)

# 获取结果
print(f"研究报告: {result['research_report_path']}")
print(f"播客脚本: {result['podcast_script_path']}")
print(f"播客音频: {result['podcast_audio_path']}")
```

---

### Deep Research Agent (深度研究智能体)

负责根据给定话题进行深度网络研究，生成结构化的研究报告。

#### 输入状态 (InputState)

```python
@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    topic: str = ""  # 研究话题
```

#### 完整状态 (State)

```python
@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
    
    # 研究工作流状态
    search_queries: List[str] = field(default_factory=list)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    crawled_content: List[Dict[str, Any]] = field(default_factory=list)
    analyzed_content: Dict[str, Any] = field(default_factory=dict)
    report_sections: Dict[str, str] = field(default_factory=dict)
    final_report: str = ""
    report_path: str = ""
    
    # 错误处理
    errors: List[str] = field(default_factory=list)
```

#### 主要工具

##### duckduckgo_search

```python
async def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]
```

使用 DuckDuckGo 搜索引擎进行网络搜索。

**参数:**
- `query` (str): 搜索查询字符串
- `max_results` (int): 最大返回结果数，默认10

**返回:**
- `List[Dict[str, Any]]`: 搜索结果列表，每个结果包含title、url、snippet、domain等字段

##### batch_crawl_urls

```python
async def batch_crawl_urls(urls: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]
```

批量并发爬取网页内容。

**参数:**
- `urls` (List[str]): 要爬取的URL列表
- `max_concurrent` (int): 最大并发数，默认5

**返回:**
- `List[Dict[str, Any]]`: 成功爬取的内容列表

##### generate_markdown_report

```python
async def generate_markdown_report(
    topic: str,
    analyzed_content: Dict[str, Any],
    sources: List[Dict[str, Any]]
) -> str
```

生成 Markdown 格式的研究报告。

**参数:**
- `topic` (str): 研究主题
- `analyzed_content` (Dict): 分析后的内容结构
- `sources` (List[Dict]): 信息来源列表

**返回:**
- `str`: Markdown 格式的报告内容

#### 使用示例

```python
from deep_research.graph import graph
from deep_research.state import InputState

# 创建输入状态
input_state = InputState(
    topic="区块链技术在供应链管理中的应用",
    messages=[]
)

# 执行研究流程
result = await graph.ainvoke(
    input_state,
    config={"configurable": {"thread_id": "blockchain_research"}}
)

# 获取研究报告
if result.get("final_report"):
    print(f"报告已生成: {result['report_path']}")
    print(f"报告内容预览: {result['final_report'][:500]}...")
```

---

### Podcast Agent (播客生成智能体)

将研究报告转化为播客内容，包括脚本生成和语音合成。

#### 输入状态 (InputState)

```python
@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    input_report: str = ""  # 输入的研究报告内容
```

#### 完整状态 (State)

```python
@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
    
    # 内容分析
    key_points: List[str] = field(default_factory=list)
    dialog_structure: Dict[str, Any] = field(default_factory=dict)
    characters: List[Character] = field(default_factory=list)
    
    # 脚本生成
    script_content: str = ""
    script_path: str = ""
    
    # 音频生成
    audio_segments: List[Dict[str, Any]] = field(default_factory=list)
    final_audio_path: str = ""
    
    # 元数据
    estimated_duration: int = 0
    
    # 错误处理
    errors: List[str] = field(default_factory=list)
```

#### 主要工具

##### analyze_research_report

```python
async def analyze_research_report(report_content: str) -> Dict[str, Any]
```

分析研究报告内容，提取关键信息点。

**参数:**
- `report_content` (str): 研究报告的 Markdown 内容

**返回:**
- `Dict[str, Any]`: 包含关键要点、话题、预估时长等信息的字典

##### design_podcast_characters

```python
async def design_podcast_characters(topic: str, key_points: List[Dict[str, Any]]) -> List[Character]
```

基于话题和内容设计播客角色。

**参数:**
- `topic` (str): 播客主题
- `key_points` (List[Dict]): 从报告中提取的关键要点

**返回:**
- `List[Character]`: 角色对象列表

##### generate_podcast_script

```python
async def generate_podcast_script(
    topic: str, 
    key_points: List[Dict[str, Any]], 
    characters: List[Character],
    target_duration: int = 1800
) -> str
```

生成播客对话脚本。

**参数:**
- `topic` (str): 播客主题
- `key_points` (List[Dict]): 关键要点
- `characters` (List[Character]): 角色列表
- `target_duration` (int): 目标时长(秒)，默认1800秒(30分钟)

**返回:**
- `str`: Markdown 格式的播客脚本

##### synthesize_speech_qwen

```python
async def synthesize_speech_qwen(
    text: str, 
    voice_model: str = "xiaoyun",
    output_path: Optional[str] = None
) -> Optional[str]
```

使用千问 TTS 服务合成语音。

**参数:**
- `text` (str): 要合成的文本内容
- `voice_model` (str): 语音模型名称，默认"xiaoyun"
- `output_path` (Optional[str]): 输出文件路径，可选

**返回:**
- `Optional[str]`: 成功时返回音频文件路径，失败时返回None

##### combine_audio_segments

```python
async def combine_audio_segments(audio_files: List[str], output_path: str) -> bool
```

合并多个音频片段为完整播客。

**参数:**
- `audio_files` (List[str]): 音频文件路径列表
- `output_path` (str): 输出文件路径

**返回:**
- `bool`: 合并成功返回True，失败返回False

#### 使用示例

```python
from podcast.graph import graph
from podcast.state import InputState

# 假设已有研究报告内容
research_report = """
# AI在医疗中的应用 - 深度研究报告

## 执行摘要
人工智能技术在医疗领域...

## 关键发现
- 医疗影像诊断准确率提升...
- 药物研发周期缩短...
"""

# 创建输入状态
input_state = InputState(
    input_report=research_report,
    messages=[]
)

# 执行播客生成流程
result = await graph.ainvoke(
    input_state,
    config={"configurable": {"thread_id": "medical_ai_podcast"}}
)

# 获取结果
if result.get("final_audio_path"):
    print(f"播客脚本: {result['script_path']}")
    print(f"播客音频: {result['final_audio_path']}")
```

---

## 数据模型

### Character (角色)

```python
@dataclass
class Character:
    name: str                           # 角色名称
    role: str                          # 角色定位 (如:"主持人", "专家")
    personality: str                   # 性格描述
    voice_config: Dict[str, Any]      # 语音配置
    background: Optional[str] = None   # 背景介绍
    expertise: Optional[List[str]] = None  # 专业领域
```

### DialogSegment (对话片段)

```python
@dataclass
class DialogSegment:
    segment_id: int                    # 片段ID
    speaker: str                       # 说话人
    content: str                       # 对话内容
    emotion: str                       # 情感色彩
    duration_estimate: int             # 预估时长(秒)
    audio_file: Optional[str] = None   # 音频文件路径
    voice_config: Optional[Dict[str, Any]] = None  # 语音配置
```

### ResearchReport (研究报告)

```python
@dataclass
class ResearchReport:
    topic: str                         # 研究主题
    summary: str                       # 摘要
    key_findings: List[str]           # 关键发现
    sections: Dict[str, str]          # 报告章节
    sources: List[str]                # 信息来源
    metadata: Dict[str, Any]          # 元数据
    created_at: datetime              # 创建时间
    file_path: Optional[str] = None   # 文件路径
```

### PodcastScript (播客脚本)

```python
@dataclass
class PodcastScript:
    title: str                        # 播客标题
    characters: List[Character]       # 角色列表
    segments: List[DialogSegment]     # 对话片段
    total_duration: int              # 总时长(秒)
    metadata: Dict[str, Any]         # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    script_path: Optional[str] = None    # 脚本文件路径
    audio_path: Optional[str] = None     # 音频文件路径
```

---

## 配置管理

### SystemConfig (系统配置)

```python
@dataclass
class SystemConfig:
    llm_config: LLMConfig              # LLM配置
    tts_config: TTSConfig              # TTS配置
    search_config: SearchConfig        # 搜索配置
    output_config: OutputConfig        # 输出配置
```

### 配置获取

```python
from shared.config import get_system_config, get_environment_config

# 获取系统配置
config = get_system_config()

# 获取环境配置
env_config = get_environment_config()

# 验证API密钥
from shared.config import validate_api_keys
validation_result = validate_api_keys()
print(validation_result)  # {"llm_api_key": True, "tts_api_key": True}
```

---

## 错误处理

### 自定义异常

```python
# 基础异常
class DeepPodcastException(Exception):
    def __init__(self, message: str, severity: ErrorSeverity, category: ErrorCategory, ...)

# 配置错误
class ConfigurationError(DeepPodcastException):
    pass

# 网络错误  
class NetworkError(DeepPodcastException):
    pass

# API错误
class APIError(DeepPodcastException):
    pass

# 解析错误
class ParsingError(DeepPodcastException):
    pass

# 验证错误
class ValidationError(DeepPodcastException):
    pass
```

### 错误处理装饰器

```python
from shared.error_handling import handle_errors, with_circuit_breaker, retry_on_network_error

@handle_errors(component="my_component", operation="my_operation")
async def my_function():
    # 自动错误处理
    pass

@with_circuit_breaker(service="external_api")
async def call_external_api():
    # 熔断器保护
    pass

@retry_on_network_error(max_attempts=3)
async def network_request():
    # 网络错误重试
    pass
```

### 安全执行

```python
from shared.error_handling import safe_execute, safe_execute_async

# 同步安全执行
result = safe_execute(risky_function, arg1, arg2, default="fallback_value")

# 异步安全执行
result = await safe_execute_async(risky_async_function, arg1, arg2, default="fallback_value")
```

---

## 工具函数

### 通用工具

```python
from shared.utils import (
    generate_task_id,
    clean_filename,
    create_topic_hash,
    generate_timestamped_filename,
    save_text_file,
    load_text_file,
    save_json_file,
    load_json_file
)

# 生成任务ID
task_id = generate_task_id()

# 清理文件名
clean_name = clean_filename("我的播客#2024", max_length=50)

# 生成时间戳文件名
filename = generate_timestamped_filename("podcast", "AI技术", ".mp3")

# 异步文件操作
await save_text_file("内容", "/path/to/file.txt")
content = await load_text_file("/path/to/file.txt")
```

### 内容处理工具

```python
from shared.utils import (
    estimate_reading_time,
    format_duration,
    split_into_chunks,
    extract_key_points,
    calculate_content_score
)

# 估算阅读时间
time_seconds = estimate_reading_time("很长的文本内容...")

# 格式化时长
formatted = format_duration(1800)  # "30分0秒"

# 分割文本块
chunks = split_into_chunks("很长的文本...", max_chunk_size=1000)

# 提取关键点
key_points = extract_key_points("文本内容...", max_points=5)

# 计算内容质量分数
score = calculate_content_score("文本内容...")  # 0-1之间的分数
```

---

## 状态枚举

```python
from shared.models import TaskStatus, ResearchDepth, PodcastStyle

# 任务状态
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# 研究深度
class ResearchDepth(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

# 播客风格
class PodcastStyle(Enum):
    CONVERSATIONAL = "conversational"
    INTERVIEW = "interview"
    NARRATIVE = "narrative"
    EDUCATIONAL = "educational"
```

---

## 最佳实践

### 1. 错误处理

```python
# 使用结构化错误处理
try:
    result = await some_operation()
except DeepPodcastException as e:
    logger.error(f"业务错误: {e}")
    # 根据错误类型和严重程度进行处理
except Exception as e:
    logger.error(f"未知错误: {e}")
    # 通用错误处理
```

### 2. 配置管理

```python
# 始终通过配置管理器获取配置
from shared.config import get_system_config

config = get_system_config()
# 使用 config.llm_config.api_key 而不是直接访问环境变量
```

### 3. 异步操作

```python
# 正确的异步模式
async def main():
    tasks = [
        process_item(item1),
        process_item(item2),
        process_item(item3)
    ]
    results = await asyncio.gather(*tasks)
```

### 4. 资源清理

```python
# 使用上下文管理器或确保资源清理
try:
    resource = acquire_resource()
    # 使用资源
finally:
    if resource:
        release_resource(resource)
```

---

## 版本信息

- **API版本**: v0.1.0
- **兼容的LangGraph版本**: >=0.6.0,<0.7.0
- **Python版本要求**: >=3.11,<4.0

---

## 更新日志

### v0.1.0 (2024-01-XX)
- 初始API版本
- 完整的三智能体架构
- 支持深度研究和播客生成
- 集成千问TTS服务
- 完善的错误处理和配置管理