@echo off
REM 深度播客系统快速启动脚本 (Windows)

echo 🚀 深度播客系统 - 快速启动
echo ================================

REM 检查 uv 是否已安装
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ UV 未安装，请先安装 UV:
    echo    pip install uv
    pause
    exit /b 1
)

echo ✅ UV 已安装
uv --version

REM 检查 Python 版本
echo 🐍 检查 Python 版本...
uv python list | findstr "3.11" >nul
if %errorlevel% neq 0 (
    echo 📦 安装 Python 3.11...
    uv python install 3.11
)

REM 创建虚拟环境并安装依赖
echo 📦 安装项目依赖...
uv sync --all-extras

REM 复制环境配置文件
if not exist .env (
    echo ⚙️  创建环境配置文件...
    copy .env.example .env
    echo 📝 请编辑 .env 文件并填入你的 API 密钥
)

REM 运行系统验证
echo 🔍 运行系统验证...
uv run python validate_system.py

echo.
echo 🎉 安装完成！
echo.
echo 📋 后续步骤：
echo 1. 编辑 .env 文件，填入 API 密钥:
echo    - ANTHROPIC_API_KEY 或 OPENAI_API_KEY
echo    - QWEN_TTS_API_KEY
echo.
echo 2. 运行测试：
echo    uv run pytest tests/ -v
echo.
echo 3. 启动 LangGraph Studio（可选）：
echo    uv run langgraph dev
echo.
echo 4. 开始使用：
echo    uv run python -c "from deep_podcast.graph import graph; print('系统就绪！')"
echo.
echo 📚 更多信息请参考：
echo    - README.md
echo    - docs\uv_guide.md
echo    - docs\api_reference.md

pause