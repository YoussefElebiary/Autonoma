import asyncio
import os
import sys

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.status import Status
from rich.traceback import install

install(show_locals=False)

from .graph.workflow import app

console = Console()

async def run_pipeline():
    console.print()
    welcome_text = Text("WELCOME TO AUTONOMA", justify="center", style="bold cyan")
    welcome_sub = Text("Autonomous ML Pipeline Orchestrator by Youssef Elebiary", justify="center", style="dim")
    welcome_panel = Panel(
        Text.assemble(welcome_text, "\n", welcome_sub),
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(welcome_panel)
    console.print()
    
    # --- THE UI / UX LAYER ---
    console.print("Please provide the path to your dataset.", style="bold")
    console.print("Example: [dim]data/titanic.csv[/dim]")
    
    try:
        csv_path = Prompt.ask("[bold green]Dataset Path[/bold green]").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold red]Operation cancelled by user.[/bold red] Exiting.")
        sys.exit(0)
    
    if not csv_path:
        console.print("[bold red]Error:[/bold red] No path provided. Exiting.")
        sys.exit(1)

    if not os.path.isfile(csv_path):
        console.print(f"[bold red]Error:[/bold red] File not found at '{csv_path}'. Please check the path and try again.")
        sys.exit(1)

    # Seed the Blackboard with the user's input
    initial_state = {
        "csv_path": csv_path,
        "critic_iterations": 0,
        "critic_feedback": "None"
    }
    
    console.print()
    
    # --- GRAPH EXECUTION ---
    try:
        final_state = {}
        with console.status("[bold cyan]Booting up LangGraph Orchestrator...", spinner="dots") as status:
            async for event in app.astream(initial_state):
                for node_name, state_update in event.items():
                    final_state.update(state_update)
                    # Clear status temporarily to print the node completion
                    console.print(f"[[bold green]OK[/bold green]] Node Finished: [bold yellow]{node_name}[/bold yellow]")
                    
                    # UX: Show what the Critic is thinking
                    if node_name == "Critic":
                        decision = state_update.get('final_decision', 'UNKNOWN').upper()
                        feedback = state_update.get('critic_feedback', '')
                        
                        decision_color = "green" if decision == "APPROVE" else "yellow"
                        
                        critic_panel = Panel(
                            f"[bold]Decision:[/bold] [{decision_color}]{decision}[/{decision_color}]\n[bold]Feedback:[/bold] {feedback}",
                            title="[bold magenta]Critic Insights[/bold magenta]",
                            border_style="magenta",
                            padding=(0, 2)
                        )
                        console.print(critic_panel)
                    
                    # UX: Show when tools are executed
                    if node_name == "Executor":
                        console.print("    [dim]>[/dim] MCP Tools executed successfully.")
                    
                    # Update status for the next node
                    status.update(f"[bold cyan]Orchestrating pipeline... Waiting on next node after {node_name}[/bold cyan]")
                    
    except KeyboardInterrupt:
        console.print("\n[bold red]Pipeline interrupted by user.[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print_exception(show_locals=False)
        console.print(f"\n[bold red]Pipeline crashed with error:[/bold red] {e}")
        sys.exit(1)

    console.print()
    
    model_path = final_state.get("model_path")
    if model_path and model_path != "No current model exists.":
        model_panel = Panel(
            f"[bold green]Model successfully saved at:[/bold green]\n{model_path}",
            title="[bold cyan]Final Output[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(model_panel)
        console.print()

    success_panel = Panel(
        "[bold green]AUTONOMA PIPELINE COMPLETE![/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    console.print(success_panel)

def main():
    try:
        asyncio.run(run_pipeline())
    except (KeyboardInterrupt, EOFError):
        # Catch any stray keyboard interrupts or EOF errors at the top level
        pass

if __name__ == "__main__":
    main()