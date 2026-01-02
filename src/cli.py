"""
命令行接口
"""

import argparse
from pathlib import Path
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Ensure the project root is on sys.path when running as a script (e.g., `uv run src/cli.py`).
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config
from src.document_loader import DocumentLoader
from src.rag_pipeline import RAGPipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="本地RAG助手")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 索引命令
    index_parser = subparsers.add_parser("index", help="索引文档")
    index_parser.add_argument("path", help="文档路径（文件或目录）")
    index_parser.add_argument("--recursive", action="store_true", help="递归处理目录")

    # 问答命令
    query_parser = subparsers.add_parser("query", help="回答问题")
    query_parser.add_argument("question", help="要回答的问题")
    query_parser.add_argument("--reasoning", action="store_true", help="显示推理过程")

    # 交互模式
    subparsers.add_parser("chat", help="交互式聊天模式")

    # 检查命令
    subparsers.add_parser("check", help="检查系统状态")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        rag = RAGPipeline()

        if args.command == "index":
            index_documents(args.path, rag, args.recursive)
        elif args.command == "query":
            answer_question(args.question, rag, args.reasoning)
        elif args.command == "chat":
            rag.interactive_session()
        elif args.command == "check":
            check_system_status(rag)

    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        sys.exit(1)


def index_documents(path: str, rag: RAGPipeline, recursive: bool):
    """索引文档"""
    path_obj = Path(path)

    if not path_obj.exists():
        console.print(f"[red]路径不存在: {path}[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("加载文档...", total=None)

        if path_obj.is_file():
            documents = [DocumentLoader.load_document(path)]
            documents = [doc for doc in documents if doc]  # 过滤None
            if documents:
                documents = DocumentLoader.chunk_document(documents[0])
        else:
            documents = DocumentLoader.load_directory(path, recursive)

        progress.update(task, completed=1, description="索引文档...")

        count = rag.index_documents(documents)

        console.print(f"[green]✅ 成功索引 {count} 个文档块[/green]")


def answer_question(question: str, rag: RAGPipeline, show_reasoning: bool):
    """回答问题"""
    result = rag.answer_question(question, use_reasoning=show_reasoning)

    table = Table(title="问答结果", show_header=False, box=None)
    table.add_column("字段", style="cyan")
    table.add_column("内容", style="white")

    table.add_row("问题", question)
    table.add_row("答案", result["answer"])

    if show_reasoning and result.get("reasoning"):
        table.add_row("推理", result["reasoning"])

    console.print(table)

    # 显示来源
    if result.get("sources"):
        console.print("\n[bold cyan]📚 参考来源:[/bold cyan]")
        for i, source in enumerate(result["sources"][:3], 1):
            content_preview = (
                source["content"][:150] + "..."
                if len(source["content"]) > 150
                else source["content"]
            )
            console.print(f"  {i}. {content_preview}")


def check_system_status(rag: RAGPipeline):
    """检查系统状态"""
    console.print(Panel.fit("[bold cyan]系统状态检查[/bold cyan]", border_style="cyan"))

    # 检查Ollama
    try:
        import ollama

        response = ollama.list()
        models = [m["name"] for m in response["models"]]
        console.print(f"✅ [green]Ollama连接正常[/green]")
        console.print(f"   可用模型: {', '.join(models)}")
        console.print(f"   当前模型: {config.ollama_model}")
    except Exception as e:
        console.print(f"❌ [red]Ollama连接失败: {e}[/red]")

    # 检查向量数据库
    try:
        count = rag.chroma_client.get_collection("documents").count()
        console.print(f"✅ [green]向量数据库正常[/green]")
        console.print(f"   已存储文档块: {count}")
    except:
        console.print(f"⚠️ [yellow]向量数据库为空或未初始化[/yellow]")

    # 检查嵌入模型
    console.print(f"✅ [green]嵌入模型已加载[/green]")
    console.print(f"   模型名称: {config.embedding_model}")


if __name__ == "__main__":
    main()
