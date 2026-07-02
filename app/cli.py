"""
CLI — DocuQuery Management Commands
=====================================
Usage:
  docuquery ingest  --source ./docs
  docuquery serve   [--host 0.0.0.0] [--port 8000]
  docuquery eval    --samples eval_data.json
  docuquery mlflow  [--port 5000]
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="docuquery",
    help="DocuQuery RAG — Next-Generation Domain-Specific RAG Pipeline CLI",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    source: str = typer.Option(..., "--source", "-s", help="Directory containing source documents"),
    strategy: str = typer.Option("late_chunking", "--strategy", help="Chunking strategy"),
) -> None:
    """Ingest documents from a directory into the vector index."""
    import os

    os.environ["CHUNK_STRATEGY"] = strategy

    from app.ingestion import IngestionPipeline
    from app.logging_config import configure_logging

    configure_logging()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=None)
        pipeline = IngestionPipeline()
        summary = pipeline.run(source)
        progress.update(task, description="✅ Ingestion complete!")

    table = Table(title="Ingestion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in summary.items():
        table.add_row(str(k), str(v))
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable hot-reload (dev only)"),
) -> None:
    """Start the FastAPI application server."""
    import uvicorn

    console.print(
        Panel.fit(
            f"[bold green]Starting DocuQuery RAG API[/bold green]\n"
            f"Host: [cyan]{host}[/cyan]   Port: [cyan]{port}[/cyan]\n"
            f"Docs: [link=http://{host}:{port}/docs]http://{host}:{port}/docs[/link]",
            title="🚀 DocuQuery",
            border_style="green",
        )
    )
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,  # structlog handles logging
    )


@app.command()
def eval(
    samples_file: str = typer.Option(..., "--samples", "-s", help="Path to JSON evaluation dataset"),
    run_name: str = typer.Option("ragas-eval", "--run-name"),
) -> None:
    """Run RAGAS evaluation on a JSON dataset."""
    from app.logging_config import configure_logging
    from app.mlops import EvalSample, run_ragas_evaluation

    configure_logging()
    path = Path(samples_file)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    data = json.loads(path.read_text())
    samples = [
        EvalSample(
            user_input=s["user_input"],
            response=s["response"],
            retrieved_contexts=s.get("retrieved_contexts", []),
            reference=s.get("reference"),
        )
        for s in data
    ]

    console.print(f"[yellow]Running RAGAS evaluation on {len(samples)} samples...[/yellow]")
    scores = run_ragas_evaluation(samples, run_name=run_name)

    table = Table(title="RAGAS Evaluation Results", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Status", style="white")

    thresholds = {
        "faithfulness": 0.90,
        "context_precision": 0.80,
        "answer_relevancy": 0.80,
        "context_recall": 0.75,
    }
    for metric, score in scores.items():
        threshold = thresholds.get(metric, 0.75)
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        table.add_row(metric, f"{score:.4f}", status)

    console.print(table)


@app.command()
def mlflow_server(
    port: int = typer.Option(5000, "--port", "-p"),
    backend: str = typer.Option("sqlite:///mlflow.db", "--backend"),
) -> None:
    """Launch a local MLflow tracking server."""
    console.print(
        Panel.fit(
            f"[bold blue]Starting MLflow Tracking Server[/bold blue]\n"
            f"Port: [cyan]{port}[/cyan]\n"
            f"UI: [link=http://localhost:{port}]http://localhost:{port}[/link]",
            title="📊 MLflow",
            border_style="blue",
        )
    )
    subprocess.run(
        ["mlflow", "server", "--backend-store-uri", backend, "--port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
