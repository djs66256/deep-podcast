# Deep Podcast System - 深度播客生成系统

[![CI](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent)

深度播客生成系统是一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 的多智能体系统，能够从用户提供的研究话题自动生成高质量的研究报告，并将其转化为生动有趣的播客音频内容。

![系统架构图](./static/system_architecture.png)

## ✨ 系统特性

### 🔬 深度研究能力
- **智能搜索**: 使用 DuckDuckGo 进行多维度网络搜索
- **内容爬取**: 并行爬取和解析网页内容
- **质量评估**: 智能评估内容质量和相关性
- **结构化分析**: 提取关键信息并生成结构化报告

### 🎙️ 播客生成能力
- **智能脚本**: 将研究报告转化为自然对话脚本
- **角色设计**: 根据话题特点设计合适的播客角色
- **语音合成**: 集成阿里千问TTS，生成高质量语音
- **音频处理**: 自动合并音频片段，生成完整播客

### 🏗️ 技术架构
- **多智能体协作**: Deep Research + Podcast + Deep Podcast Controller
- **异步处理**: 高效的并发处理能力
- **错误恢复**: 完善的错误处理和重试机制
- **配置管理**: 灵活的环境配置和参数调优

## 🚀 快速开始

### 前置要求

- Python 3.11+
- [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio) (可选)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd deep-podcast-system
```

2. **安装 UV（推荐）**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip
pip install uv
```

3. **使用 UV 管理环境**
```bash
# 创建虚拟环境并安装所有依赖
uv sync

# 安装开发依赖
uv sync --all-extras

# 激活环境（可选，uv run 会自动使用虚拟环境）
source .venv/bin/activate
```

4. **传统方式（pip + venv）**
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

5. **环境配置**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

6. **运行测试**
```bash
# 使用 uv
uv run pytest tests/ -v

# 或者传统方式
pytest tests/ -v
```

### 环境配置

创建 `.env` 文件并配置以下变量：

```bash
# LLM 配置 (选择一个)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# 或者
OPENAI_API_KEY=your_openai_api_key_here

# TTS 配置 (阿里千问)
QWEN_TTS_API_KEY=your_qwen_tts_api_key_here

# 可选：搜索增强
TAVILY_API_KEY=your_tavily_api_key_here
```

## 📖 使用指南

### 在 LangGraph Studio 中使用

1. 打开 LangGraph Studio
2. 加载项目目录
3. 选择 `deep_podcast` 图
4. 输入研究话题并运行

### 命令行使用

```python
from deep_podcast.graph import graph
from deep_podcast.state import InputState

# 创建输入状态
input_state = InputState(
    user_topic="人工智能发展趋势",
    messages=[]
)

# 执行完整流程
result = await graph.ainvoke(
    input_state,
    config={"configurable": {"thread_id": "my_podcast"}}
)

# 获取结果
print(f"研究报告: {result['research_report_path']}")
print(f"播客脚本: {result['podcast_script_path']}")
print(f"播客音频: {result['podcast_audio_path']}")
```

### 单独使用子系统

#### 深度研究
```python
from deep_research.graph import graph as research_graph
from deep_research.state import InputState as ResearchInput

result = await research_graph.ainvoke(
    ResearchInput(topic="区块链技术应用"),
    config={"configurable": {"thread_id": "research_001"}}
)
```

#### 播客生成
```python
from podcast.graph import graph as podcast_graph
from podcast.state import InputState as PodcastInput

result = await podcast_graph.ainvoke(
    PodcastInput(input_report=research_report_content),
    config={"configurable": {"thread_id": "podcast_001"}}
)
```

## 🏛️ 系统架构

### 核心组件

```
深度播客系统
├── deep_research/          # 深度研究智能体
│   ├── tools.py           # 搜索、爬虫、分析工具
│   ├── graph.py           # 研究工作流图
│   └── state.py           # 研究状态管理
├── podcast/               # 播客生成智能体
│   ├── tools.py           # 脚本生成、TTS工具
│   ├── graph.py           # 播客生成工作流图
│   └── state.py           # 播客状态管理
├── deep_podcast/          # 主控制器
│   ├── graph.py           # 整体协调工作流
│   └── state.py           # 全局状态管理
└── shared/                # 共享组件
    ├── models.py          # 数据模型
    ├── config.py          # 配置管理
    ├── utils.py           # 工具函数
    └── error_handling.py  # 错误处理
```

### 工作流程

1. **话题分析**: 分析用户输入的研究话题
2. **深度研究**: 执行网络搜索、内容爬取、信息分析
3. **报告生成**: 生成结构化的Markdown研究报告
4. **内容转换**: 分析报告内容，提取关键信息
5. **角色设计**: 根据话题特点设计播客角色
6. **脚本生成**: 生成自然流畅的播客对话脚本
7. **语音合成**: 使用TTS技术生成各角色的语音
8. **音频合成**: 合并音频片段，生成最终播客

## ⚙️ 配置选项

### LLM 配置

支持多种LLM提供商：
- **Anthropic Claude**: 推荐用于复杂推理任务
- **OpenAI GPT**: 兼容各种OpenAI模型
- **自定义服务**: 支持OpenAI兼容的自定义API

### TTS 配置

- **阿里千问TTS**: 高质量中文语音合成
- **多角色语音**: 支持不同角色使用不同声音
- **语音参数**: 可调节语速、音调、音量

### 搜索配置

- **搜索结果数量**: 可配置最大搜索结果数
- **爬取页面数**: 可配置最大爬取页面数
- **内容质量过滤**: 自动过滤低质量内容
- **并发控制**: 可配置并发请求数量

## 🧪 测试

### 运行测试套件

```bash
# 使用 uv 运行测试
uv run pytest                              # 运行所有测试
uv run pytest tests/unit_tests/ -v         # 运行单元测试
uv run pytest tests/integration_tests/ -v -m integration  # 运行集成测试
uv run pytest tests/ -v -m performance     # 运行性能测试

# 运行真实服务测试 (需要API密钥)
RUN_INTEGRATION_TESTS=1 uv run pytest tests/ -v -m real_services

# 传统方式（在激活的虚拟环境中）
pytest                                     # 运行所有测试
pytest tests/unit_tests/ -v               # 运行单元测试
pytest tests/integration_tests/ -v -m integration  # 运行集成测试
```

### 测试覆盖率

```bash
# 使用 uv
uv run pytest --cov=src --cov-report=html

# 传统方式
pytest --cov=src --cov-report=html
```

## 📊 性能优化

### 建议配置

**开发环境**:
```bash
MAX_SEARCH_RESULTS=10
MAX_CRAWL_PAGES=20
CONCURRENT_REQUESTS=3
```

**生产环境**:
```bash
MAX_SEARCH_RESULTS=20
MAX_CRAWL_PAGES=50
CONCURRENT_REQUESTS=5
```

### 性能监控

系统内置性能监控和错误追踪：
- 请求延迟监控
- 错误率统计
- 资源使用监控
- 任务完成率追踪

## 🔧 故障排除

### 常见问题

**API密钥错误**
```bash
# 检查环境变量
echo $ANTHROPIC_API_KEY
echo $QWEN_TTS_API_KEY
```

**网络连接问题**
```bash
# 检查网络连接
curl -I https://api.anthropic.com
curl -I https://dashscope.aliyuncs.com
```

**依赖问题**
```bash
# 重新安装依赖
pip install --force-reinstall -e .
```

### 日志调试

启用详细日志：
```bash
LOG_LEVEL=DEBUG python your_script.py
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 开发规范

- 遵循 PEP 8 代码风格
- 添加适当的类型注解
- 编写全面的测试用例
- 更新相关文档

## 📝 更新日志

### v0.1.0 (2024-01-XX)
- ✨ 初始版本发布
- 🔬 完整的深度研究功能
- 🎙️ 播客生成和语音合成
- 🏗️ 多智能体架构
- 🧪 完整的测试套件

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [LangGraph](https://github.com/langchain-ai/langgraph) - 多智能体工作流框架
- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用开发框架
- [阿里千问](https://dashscope.aliyun.com/) - 高质量TTS服务
- [DuckDuckGo](https://duckduckgo.com/) - 隐私友好的搜索服务

## 📞 支持

如果你在使用过程中遇到问题：

1. 查看 [常见问题](docs/faq.md)
2. 搜索 [Issues](../../issues)
3. 创建新的 [Issue](../../issues/new)
4. 查看 [讨论区](../../discussions)

---

**深度播客生成系统** - 让AI为你创造有价值的内容

*Built with ❤️ by the Deep Podcast Team*
