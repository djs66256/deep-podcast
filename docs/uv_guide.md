# UV 虚拟环境管理指南

## 安装 UV

首先安装 uv（如果还没有安装）：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip 安装
pip install uv

# 验证安装
uv --version
```

## 项目设置

### 1. 初始化项目虚拟环境

```bash
# 克隆项目后，进入项目目录
cd deep-podcast-system

# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 或者指定 Python 版本
uv python install 3.11
uv sync --python 3.11
```

### 2. 激活虚拟环境

```bash
# 激活虚拟环境
source .venv/bin/activate

# 或者使用 uv 运行命令（无需手动激活）
uv run python your_script.py
```

### 3. 安装开发依赖

```bash
# 安装所有依赖（包括开发依赖）
uv sync --all-extras

# 仅安装生产依赖
uv sync --no-dev

# 安装特定分组的依赖
uv sync --group dev
```

## 日常开发工作流

### 添加新依赖

```bash
# 添加生产依赖
uv add requests>=2.31.0

# 添加开发依赖
uv add --dev pytest-mock

# 添加可选依赖
uv add --optional audio pydub

# 从 requirements.txt 添加
uv add -r requirements.txt
```

### 移除依赖

```bash
# 移除依赖
uv remove requests

# 移除开发依赖
uv remove --dev pytest-mock
```

### 运行命令

```bash
# 运行 Python 脚本
uv run python validate_system.py

# 运行测试
uv run pytest tests/ -v

# 运行 lint 检查
uv run ruff check src/

# 运行类型检查
uv run mypy src/
```

### 查看依赖

```bash
# 查看所有依赖
uv pip list

# 查看依赖树
uv pip tree

# 查看过期的依赖
uv pip list --outdated
```

## 环境管理

### 创建多个环境

```bash
# 为不同目的创建环境
uv venv .venv-dev --python 3.11
uv venv .venv-prod --python 3.11

# 切换环境
export VIRTUAL_ENV=".venv-dev"
uv sync
```

### 环境清理

```bash
# 清理缓存
uv cache clean

# 重新创建环境
rm -rf .venv
uv sync
```

## CI/CD 集成

### GitHub Actions 示例

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
        with:
          version: "latest"
      
      - name: Set up Python
        run: uv python install 3.11
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run tests
        run: uv run pytest tests/ -v
```

### Docker 集成

```dockerfile
FROM python:3.11-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 复制项目文件
COPY . /app
WORKDIR /app

# 安装依赖
RUN uv sync --frozen --no-dev

# 运行应用
CMD ["uv", "run", "python", "main.py"]
```

## 配置文件

### pyproject.toml 配置

项目已经配置了以下 uv 设置：

```toml
[tool.uv]
python = ">=3.11"
index-url = "https://pypi.org/simple"
extra-index-url = ["https://pypi.tuna.tsinghua.edu.cn/simple"]

[dependency-groups]
dev = [
    "pytest>=8.3.5",
    "pytest-asyncio>=0.21.0", 
    "mypy>=1.11.1",
    "ruff>=0.6.1",
    # ... 其他开发依赖
]
```

## 常用命令快速参考

```bash
# 项目初始化
uv sync                          # 安装所有依赖
uv sync --frozen                 # 使用锁定版本安装
uv sync --no-dev                 # 仅生产依赖

# 依赖管理
uv add package                   # 添加依赖
uv add --dev package            # 添加开发依赖
uv remove package               # 移除依赖
uv lock                         # 更新锁文件

# 运行命令
uv run command                  # 在虚拟环境中运行命令
uv run python script.py        # 运行 Python 脚本
uv run pytest                  # 运行测试

# 环境管理
uv venv                         # 创建虚拟环境
uv pip list                     # 查看已安装包
uv cache clean                  # 清理缓存
```

## 故障排除

### 常见问题

1. **依赖冲突**
```bash
uv lock --upgrade               # 升级所有依赖
uv sync --refresh               # 刷新环境
```

2. **缓存问题**
```bash
uv cache clean                  # 清理缓存
rm uv.lock                      # 删除锁文件重新生成
uv sync
```

3. **Python 版本问题**
```bash
uv python list                  # 查看可用 Python 版本
uv python install 3.11         # 安装特定版本
uv sync --python 3.11          # 使用特定版本
```

## 性能优化

UV 的主要优势：

- **极快的依赖解析**：比 pip 快 10-100 倍
- **并行安装**：同时下载和安装多个包
- **智能缓存**：避免重复下载
- **跨平台兼容**：支持 Windows、macOS、Linux

## 迁移指南

### 从 pip/virtualenv 迁移

```bash
# 1. 安装 uv
pip install uv

# 2. 生成 pyproject.toml（如果没有）
uv init --package

# 3. 从 requirements.txt 迁移
uv add -r requirements.txt

# 4. 创建环境并同步
uv sync
```

### 从 conda 迁移

```bash
# 1. 导出 conda 环境
conda env export > environment.yml

# 2. 手动转换依赖到 pyproject.toml
# 3. 使用 uv 重新创建环境
uv sync
```

使用 uv 可以显著提升依赖管理和环境创建的速度，特别适合大型项目的开发工作流。