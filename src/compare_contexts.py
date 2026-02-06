#!/usr/bin/env python3
"""
Compare simple vs LLM-based context generation for sample documents.

Generates both versions side-by-side to evaluate quality and estimate costs.
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Add parent directory to path to import from chunk_documents
sys.path.insert(0, str(Path(__file__).parent))

from chunk_documents import (
    load_tokenizer,
    parse_markdown,
    chunk_document,
    count_tokens,
    FORMATTED_DIR
)
from contextual_utils import get_anthropic_client, ContextGenerationStats


console = Console()


def list_sample_files(n: int = 10, source: str = None) -> list[Path]:
    """
    List sample markdown files for testing.

    Args:
        n: Number of files to return
        source: "substack", "sevan", or None (both)

    Returns:
        List of markdown file paths
    """
    md_files = []

    if source is None or source == "substack":
        substack_dir = FORMATTED_DIR / "substack"
        if substack_dir.exists():
            md_files.extend(list(substack_dir.glob("*.md")))

    if source is None or source == "sevan":
        sevan_dir = FORMATTED_DIR / "sevan"
        if sevan_dir.exists():
            md_files.extend(list(sevan_dir.glob("*.md")))

    return md_files[:n]


def compare_contexts(file_paths: list[Path], show_full: bool = False):
    """
    Compare simple vs LLM context for sample documents.

    Args:
        file_paths: List of markdown files to process
        show_full: If True, show full chunk text; otherwise show first 300 chars
    """
    tokenizer = load_tokenizer()
    anthropic_client = get_anthropic_client()

    if not anthropic_client:
        console.print("[red]Error: ANTHROPIC_API_KEY not found in environment[/red]")
        console.print("Please set ANTHROPIC_API_KEY to use LLM context mode")
        return

    stats_tracker = ContextGenerationStats()

    console.print("\n[bold cyan]═" * 60 + "[/bold cyan]")
    console.print("[bold cyan]Context Comparison: Simple vs LLM[/bold cyan]")
    console.print("[bold cyan]═" * 60 + "[/bold cyan]\n")

    total_simple_chunks = 0
    total_llm_chunks = 0

    for file_idx, file_path in enumerate(file_paths, 1):
        if not file_path.exists():
            console.print(f"[yellow]File not found: {file_path}[/yellow]")
            continue

        console.print(f"\n[bold]Document {file_idx}/{len(file_paths)}[/bold]")
        console.print(f"[dim]File: {file_path.name}[/dim]\n")

        # Parse document
        metadata = parse_markdown(file_path)

        # Generate both versions
        try:
            simple_chunks = chunk_document(
                metadata,
                tokenizer,
                enrich_context=True,
                context_mode="simple"
            )
            total_simple_chunks += len(simple_chunks)

            llm_chunks = chunk_document(
                metadata,
                tokenizer,
                enrich_context=True,
                context_mode="llm",
                anthropic_client=anthropic_client,
                stats_tracker=stats_tracker
            )
            total_llm_chunks += len(llm_chunks)

        except Exception as e:
            console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
            continue

        # Display document metadata
        meta_table = Table(show_header=False, box=None, padding=(0, 2))
        meta_table.add_column(style="cyan")
        meta_table.add_column()

        meta_table.add_row("Title:", metadata.get("title") or "[dim]No title[/dim]")
        meta_table.add_row("Date:", metadata.get("date") or "[dim]No date[/dim]")
        meta_table.add_row("Keywords:", metadata.get("keywords") or "[dim]No keywords[/dim]")
        meta_table.add_row("Chunks:", f"{len(simple_chunks)} (simple), {len(llm_chunks)} (llm)")

        console.print(meta_table)

        # Compare first 3 chunks (or fewer if document is short)
        num_to_show = min(3, len(simple_chunks), len(llm_chunks))
        console.print(f"\n[dim]Showing first {num_to_show} chunks:[/dim]\n")

        for i in range(num_to_show):
            console.print(f"[bold yellow]━━━ Chunk {i} ━━━[/bold yellow]\n")

            simple_chunk = simple_chunks[i]
            llm_chunk = llm_chunks[i]

            # Extract context parts
            simple_text_for_embed = simple_chunk["text_for_embedding"]
            llm_text_for_embed = llm_chunk["text_for_embedding"]
            raw_text = simple_chunk["text"]

            # For simple mode, the context is the prefix before the raw text
            simple_context = simple_text_for_embed.replace(raw_text, "").strip()

            # For LLM mode, the context is stored separately
            llm_context = llm_chunk.get("llm_context", "")

            # Truncate for display
            if not show_full:
                raw_text = raw_text[:300] + "..." if len(raw_text) > 300 else raw_text

            # Display
            console.print("[cyan]Raw Chunk Text:[/cyan]")
            console.print(Panel(raw_text, border_style="dim"))

            console.print("\n[green]Simple Context:[/green]")
            console.print(Panel(simple_context or "[dim]No context[/dim]", border_style="green"))

            console.print("\n[magenta]LLM Context:[/magenta]")
            console.print(Panel(llm_context or "[dim]No context[/dim]", border_style="magenta"))

            # Token counts
            simple_tokens = count_tokens(simple_text_for_embed, tokenizer)
            llm_tokens = count_tokens(llm_text_for_embed, tokenizer)

            console.print(f"\n[dim]Tokens: Simple={simple_tokens}, LLM={llm_tokens}[/dim]\n")

        console.print("[dim]" + "─" * 60 + "[/dim]\n")

    # Summary statistics
    console.print("\n[bold cyan]═" * 60 + "[/bold cyan]")
    console.print("[bold cyan]Summary Statistics[/bold cyan]")
    console.print("[bold cyan]═" * 60 + "[/bold cyan]\n")

    summary_table = Table(show_header=True, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Documents processed", str(len(file_paths)))
    summary_table.add_row("Total chunks (simple)", str(total_simple_chunks))
    summary_table.add_row("Total chunks (LLM)", str(total_llm_chunks))

    if stats_tracker.num_requests > 0:
        summary = stats_tracker.summary()
        summary_table.add_row("LLM API calls", str(summary['requests']))
        summary_table.add_row("Cache hit rate", f"{summary['cache_hit_rate']:.1%}")
        summary_table.add_row("Cost (this sample)", f"${summary['cost_usd']:.4f}")

        # Estimate cost for full corpus
        if total_llm_chunks > 0:
            # Estimate: assume ~6,850 total chunks (from existing data)
            estimated_full_cost = (6850 / total_llm_chunks) * summary['cost_usd']
            summary_table.add_row(
                "[bold]Estimated cost (full corpus)[/bold]",
                f"[bold]${estimated_full_cost:.2f}[/bold]"
            )

    console.print(summary_table)
    console.print()

    # Quality assessment guide
    console.print("[bold green]Quality Assessment Checklist:[/bold green]")
    console.print("  [green]✓[/green] Is LLM context in Turkish?")
    console.print("  [green]✓[/green] Is it accurate and relevant?")
    console.print("  [green]✓[/green] Does it add semantic value beyond metadata?")
    console.print("  [green]✓[/green] Is it concise (not just copying the chunk)?")
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare simple vs LLM-based context generation"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample documents to process (default: 5)"
    )
    parser.add_argument(
        "--source",
        choices=["substack", "sevan"],
        help="Source to sample from (default: both)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full chunk text (default: truncate to 300 chars)"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to compare (overrides --sample)"
    )

    args = parser.parse_args()

    # Get files to process
    if args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        file_paths = list_sample_files(n=args.sample, source=args.source)

    if not file_paths:
        console.print("[red]No files found to process[/red]")
        return

    console.print(f"[green]Processing {len(file_paths)} documents...[/green]")

    # Run comparison
    compare_contexts(file_paths, show_full=args.full)


if __name__ == "__main__":
    main()
