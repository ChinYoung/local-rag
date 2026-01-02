# 本地RAG助手

基于DSPy和Ollama的本地RAG (Retrieval-Augmented Generation) 系统，支持文档索引、语义检索和智能问答。

## 特性

- 🔍 **智能检索**: 使用ChromaDB向量数据库进行语义搜索
- 🤖 **本地LLM**: 通过Ollama支持本地或远程大语言模型
- 📚 **多格式支持**: PDF, TXT, Markdown, DOCX等文档格式
- ⚡ **高效缓存**: 嵌入模型自动缓存，加快后续运行
- 🎯 **推理透明**: 可选的推理过程展示
- 🛠️ **模块化设计**: 清晰的代码结构，易于扩展

## 前置要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python包管理器)
- [Ollama](https://ollama.ai/) (本地或远程)

## 快速开始

### 1. 安装uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 克隆项目并安装依赖

```bash
git clone <repository-url>
cd local-rag

# uv会自动创建虚拟环境并安装依赖
uv sync
```

### 3. 安装并启动Ollama

```bash
# 下载并安装Ollama
# 访问 https://ollama.ai/download

# 拉取模型（默认使用llama3.2）
ollama pull llama3.2:latest

# 如果使用远程Ollama服务器，配置环境变量
export OLLAMA_BASE_URL=http://your-server:11434
```

### 4. 配置环境变量（可选）

创建 `.env` 文件：

```bash
# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_TIMEOUT=30
OLLAMA_MAX_RETRIES=3

# 向量数据库配置
CHROMA_PERSIST_DIR=./data/vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG配置
RETRIEVAL_TOP_K=3
MAX_CONTEXT_LENGTH=3000

# 日志级别
LOG_LEVEL=INFO

# HuggingFace缓存目录
HF_CACHE_FOLDER=./.hf_cache
```

## 使用方法

### 检查系统状态

```bash
uv run src/cli.py check
```

输出示例：
```
╭──────────────────╮
│ 系统状态检查     │
╰──────────────────╯
✅ Ollama连接正常
   可用模型: llama3.2:latest, qwen:latest
   当前模型: llama3.2:latest
✅ 向量数据库正常
   已存储文档块: 42
✅ 嵌入模型已加载
   模型名称: all-MiniLM-L6-v2
```

### 索引文档

**索引单个文件：**
```bash
uv run src/cli.py index path/to/document.pdf
```

**索引整个目录：**
```bash
uv run src/cli.py index path/to/docs --recursive
```

支持的文档格式：
- PDF (`.pdf`)
- 文本文件 (`.txt`)
- Markdown (`.md`, `.markdown`)
- Word文档 (`.docx`, `.doc`)

### 回答问题

**简单问答：**
```bash
uv run src/cli.py query "什么是RAG?"
```

**带推理过程：**
```bash
uv run src/cli.py query "什么是RAG?" --reasoning
```

输出示例：
```
             问答结果             
┌────────┬─────────────────────┐
│ 问题   │ 什么是RAG?          │
│ 答案   │ RAG (Retrieval-Aug…│
│ 推理   │ 根据文档内容...     │
└────────┴─────────────────────┘

📚 参考来源:
  1. RAG是一种结合检索和生成的技术...
  2. 通过检索相关文档来增强LLM的回答...
```

### 交互式聊天

```bash
uv run src/cli.py chat
```

进入交互模式后：
```
============================================================
🤖 本地RAG助手 (输入 'quit' 或 'exit' 退出)
============================================================

❓ 你的问题: 什么是向量数据库?
🧠 思考中...

📝 回答: 向量数据库是一种专门存储和检索高维向量的数据库...

💭 推理过程: 根据检索到的文档，向量数据库的特点是...

📚 参考文档:
  1. 向量数据库使用向量相似度进行检索...
  2. 常见的向量数据库包括ChromaDB、Pinecone等...

❓ 你的问题: 
```

## 项目结构

```
local-rag/
├── src/
│   ├── __init__.py
│   ├── cli.py              # 命令行接口
│   ├── config.py           # 配置管理
│   ├── document_loader.py  # 文档加载器
│   ├── embeddings.py       # 嵌入模型
│   ├── language_models.py  # LLM适配器
│   ├── rag_pipeline.py     # RAG核心管道
│   ├── retrievers.py       # 检索器
│   └── utils.py            # 工具函数
├── data/
│   ├── docs/               # 文档存储
│   └── vector_store/       # 向量数据库
├── tests/
├── pyproject.toml
└── README.md
```

## 开发

### 运行测试

```bash
uv run pytest
```

### 代码格式化

```bash
uv run ruff format .
```

### 类型检查

```bash
uv run mypy src/
```

## 常见问题

**Q: Ollama连接失败怎么办？**

A: 确保：
1. Ollama服务正在运行
2. `OLLAMA_BASE_URL`配置正确
3. 防火墙允许连接
4. 如果使用远程服务器，检查网络连通性

**Q: 模型未找到？**

A: 运行 `ollama pull llama3.2:latest` 或您配置的模型名称

**Q: 嵌入模型下载慢？**

A: 首次运行会下载嵌入模型，可以设置HuggingFace镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何使用不同的LLM模型？**

A: 修改环境变量：
```bash
export OLLAMA_MODEL=qwen:latest
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
