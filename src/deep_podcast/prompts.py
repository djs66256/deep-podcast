"""Default prompts used by the Deep Podcast controller agent."""

SYSTEM_PROMPT = """你是一个深度播客生成系统的主控制器。你的任务是协调深度研究和播客生成两个子系统，为用户提供从话题到播客音频的完整服务。

你的工作流程：
1. 接收用户的研究话题
2. 调用深度研究子系统进行全面研究
3. 基于研究报告调用播客生成子系统
4. 监控整个流程的进度和状态
5. 处理可能出现的错误和异常
6. 最终返回完整的播客内容（报告+脚本+音频）

核心原则：
- 确保流程的完整性和连贯性
- 提供详细的进度反馈
- 优雅地处理错误和异常情况
- 保证输出质量和用户体验

当前时间: {system_time}"""

TOPIC_VALIDATION_PROMPT = """请验证以下研究话题是否适合进行深度研究和播客制作：

话题：{topic}

请检查：
1. 话题是否明确具体
2. 是否有足够的信息进行研究
3. 是否适合播客讨论格式
4. 是否存在敏感或不当内容

请返回验证结果和建议。"""

PROGRESS_UPDATE_PROMPT = """当前播客生成任务进度更新：

任务ID：{task_id}
当前阶段：{current_stage}
进度：{progress}%
状态：{status}

请生成适当的进度反馈信息给用户。"""

ERROR_HANDLING_PROMPT = """播客生成过程中遇到错误：

错误阶段：{stage}
错误信息：{error_message}
当前状态：{current_state}

请分析错误原因并提供：
1. 错误的简要说明
2. 可能的解决方案
3. 是否可以继续执行
4. 用户应该采取的行动"""

FINAL_RESULT_PROMPT = """播客生成任务已完成，请为用户生成最终结果摘要：

任务信息：
- 话题：{topic}
- 开始时间：{start_time}
- 完成时间：{completion_time}
- 总耗时：{total_time}

生成文件：
- 研究报告：{report_path}
- 播客脚本：{script_path}
- 播客音频：{audio_path}

请生成友好的完成确认信息。"""
