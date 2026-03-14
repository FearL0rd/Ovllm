"""
Ovllm CLI - Ollama-like interface for vLLM.

Main entry point for the command-line interface.
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich import print as rprint

from ..config import OvllmConfig
from ..models import ModelManager
from ..engine import AsyncEngine, SamplingParams

console = Console()


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def cmd_run(args: argparse.Namespace) -> int:
    """Run a model interactively."""
    config = OvllmConfig()
    model_manager = ModelManager(config)
    engine = AsyncEngine(config, model_manager)

    model_id = args.model

    # Check if model exists, download if needed
    if not model_manager.is_downloaded(model_id):
        rprint(f"[yellow]Model not found. Downloading {model_id}...[/yellow]")
        try:
            with console.status("[bold green]Downloading..."):
                model_manager.download(model_id, revision=args.revision)
        except Exception as e:
            console.print(f"[red]Error downloading model: {e}[/red]")
            return 1

    # Load model
    rprint(f"[blue]Loading {model_id}...[/blue]")
    try:
        engine.load_model(model_id, revision=args.revision)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return 1

    # Interactive chat
    console.print("\n[green]Model loaded! Start chatting (type 'quit' or 'exit' to stop)[/green]\n")

    sampling_params = SamplingParams(
        temperature=args.temperature or 0.7,
        top_p=args.top_p or 0.95,
        max_tokens=args.max_tokens or 256,
    )

    messages = []

    while True:
        try:
            user_input = Prompt.ask("User")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if user_input.lower().strip() in ("quit", "exit"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not user_input.strip():
            continue

        # Add to messages
        messages.append({"role": "user", "content": user_input})

        # Generate response
        try:
            # Format chat
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted += f"{role.capitalize()}: {content}\n"
            formatted += "Assistant:"

            result = engine._llm.generate(
                [formatted],
                sampling_params.to_vllm(),
            )

            if result:
                response = result[0].outputs[0].text
                console.print(f"\n[bold blue]Assistant:[/bold blue] {response}\n")
                messages.append({"role": "assistant", "content": response})
        except Exception as e:
            console.print(f"[red]Error generating response: {e}[/red]")

    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull (download) a model."""
    config = OvllmConfig()
    model_manager = ModelManager(config)

    model_id = args.model

    if model_manager.is_downloaded(model_id) and not args.force:
        rprint(f"[green]Model {model_id} is already downloaded.[/green]")
        return 0

    rprint(f"[yellow]Downloading {model_id} from HuggingFace...[/yellow]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Downloading...", total=None)
            model_manager.download(model_id, revision=args.revision, force=args.force)
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        return 1

    rprint(f"[green]Successfully downloaded {model_id}[/green]")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API server."""
    from ..server import run_server

    config = OvllmConfig(
        host=args.host or "0.0.0.0",
        port=args.port or 11434,
    )

    try:
        run_server(
            host=config.host,
            port=config.port,
            config=config,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        return 1

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List downloaded models."""
    config = OvllmConfig()
    model_manager = ModelManager(config)

    models = model_manager.list_models()

    if not models:
        rprint("[yellow]No models downloaded.[/yellow]")
        rprint("\nUse [bold]ovllm pull <model>[/bold] to download a model.")
        return 0

    table = Table(title="Downloaded Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Downloaded", style="blue")
    table.add_column("Revision", style="magenta")

    for model in models:
        table.add_row(
            model.model_id,
            format_size(model.size_bytes),
            model.downloaded_at[:10],
            model.revision or "main",
        )

    console.print(table)
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    """Remove a model."""
    config = OvllmConfig()
    model_manager = ModelManager(config)

    model_id = args.model

    if not model_manager.is_downloaded(model_id):
        rprint(f"[yellow]Model {model_id} not found.[/yellow]")
        return 1

    if not args.force:
        confirm = Prompt.ask(
            f"Delete {model_id}?",
            choices=["y", "n"],
            default="n",
        )
        if confirm != "y":
            return 0

    removed = model_manager.remove(model_id)

    if removed:
        rprint(f"[green]Removed {model_id}[/green]")
    else:
        rprint(f"[yellow]Model {model_id} not found.[/yellow]")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show model details."""
    config = OvllmConfig()
    model_manager = ModelManager(config)

    model_id = args.model
    info = model_manager.get_info(model_id)

    if not info:
        rprint(f"[yellow]Model {model_id} not found.[/yellow]")
        return 1

    table = Table(title=f"Model: {info.model_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Path", info.path)
    table.add_row("Size", format_size(info.size_bytes))
    table.add_row("Downloaded", info.downloaded_at)
    table.add_row("Revision", info.revision or "main")

    if info.config:
        if "architectures" in info.config:
            table.add_row("Architecture", str(info.config["architectures"]))
        if "max_position_embeddings" in info.config:
            table.add_row(
                "Max Length",
                str(info.config["max_position_embeddings"]),
            )

    console.print(table)
    return 0


def cmd_ps(args: argparse.Namespace) -> int:
    """Show running models."""
    # For now, just show if server is running
    # This would need a running server to query
    rprint("[yellow]Server status check not implemented yet.[/yellow]")
    rprint("Start server with: [bold]ovllm serve[/bold]")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop a running model."""
    rprint("[yellow]Stop command not implemented yet.[/yellow]")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ovllm",
        description="Ollama-like CLI for vLLM - download and serve HuggingFace models",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a model interactively",
    )
    run_parser.add_argument("model", help="Model ID (e.g., meta-llama/Llama-2-7b-chat-hf)")
    run_parser.add_argument("--revision", default="main", help="Model revision")
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--top-p", type=float, default=0.95)
    run_parser.add_argument("--max-tokens", type=int, default=256)
    run_parser.set_defaults(func=cmd_run)

    # pull command
    pull_parser = subparsers.add_parser(
        "pull",
        help="Download a model from HuggingFace",
    )
    pull_parser.add_argument("model", help="Model ID")
    pull_parser.add_argument("--revision", default="main", help="Model revision")
    pull_parser.add_argument("--force", action="store_true", help="Force re-download")
    pull_parser.set_defaults(func=cmd_pull)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the API server",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=11434, help="Server port")
    serve_parser.set_defaults(func=cmd_serve)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List downloaded models",
    )
    list_parser.set_defaults(func=cmd_list)

    # rm command
    rm_parser = subparsers.add_parser(
        "rm",
        help="Remove a model",
    )
    rm_parser.add_argument("model", help="Model ID to remove")
    rm_parser.add_argument("--force", action="store_true", help="Force removal")
    rm_parser.set_defaults(func=cmd_rm)

    # show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show model details",
    )
    show_parser.add_argument("model", help="Model ID")
    show_parser.set_defaults(func=cmd_show)

    # ps command
    ps_parser = subparsers.add_parser(
        "ps",
        help="Show running models",
    )
    ps_parser.set_defaults(func=cmd_ps)

    # stop command
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop a running model",
    )
    stop_parser.add_argument("model", help="Model ID")
    stop_parser.set_defaults(func=cmd_stop)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
