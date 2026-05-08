import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install

install(show_locals=False)

from autonoma.graph.workflow import app
from autonoma.config import MAX_CRITIC_ITERATIONS

console = Console()

async def run_pipeline():
    import time
    start_time = time.time()
    console.print()
    welcome_text = Text("WELCOME TO AUTONOMA", justify="center", style="bold cyan")
    welcome_sub = Text("Autonomous ML Pipeline by Youssef Elebiary", justify="center", style="dim")
    welcome_panel = Panel(
        Text.assemble(welcome_text, "\n", welcome_sub),
        border_style="cyan",
        padding=(1, 2),
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

    initial_state = {
        "csv_path": csv_path,
        "critic_iterations": 0,
        "critic_feedback": "None"
    }
    
    console.print()

    # --- GRAPH EXECUTION ---
    server_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "server.py"))
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script]
    )

    try:
        final_state = {}
        with console.status("[bold cyan]Booting up LangGraph Orchestrator...", spinner="dots") as status:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    initial_state["mcp_session"] = session
                    
                    async for event in app.astream(initial_state):
                        for node_name, state_update in event.items():
                            if state_update is not None:
                                final_state.update(state_update)
                            console.print(f"[[bold green]OK[/bold green]] Node Finished: [bold yellow]{node_name}[/bold yellow]")
                            
                            if node_name == "Critic":
                                decision = state_update.get('final_decision', 'UNKNOWN').upper()
                                feedback = state_update.get('critic_feedback', '')
                                iters = state_update.get('critic_iterations', 0)
                                
                                decision_color = "green" if decision == "APPROVE" else "yellow"
                                
                                display_decision = decision
                                if decision == "REVISE" and iters > 1:
                                    display_decision = f"{decision} ({iters}/{MAX_CRITIC_ITERATIONS})"
                                
                                critic_panel = Panel(
                                    f"[bold]Decision:[/bold] [{decision_color}]{display_decision}[/{decision_color}]\n[bold]Feedback:[/bold] {feedback}",
                                    title="[bold magenta]Critic Insights[/bold magenta]",
                                    border_style="magenta",
                                    padding=(0, 2)
                                )
                                console.print(critic_panel)
                            
                            if node_name == "Executor":
                                console.print("    [dim]>[/dim] MCP Tools executed successfully.")
                            
                        status.update(f"[bold cyan]Orchestrating pipeline... Waiting on next node after {node_name}[/bold cyan]")

                    # --- FINAL REPORTING (Inside Session) ---
                    console.print()
                    
                    model_path = final_state.get("model_path")
                    metrics = final_state.get("evaluation_metrics")
                    params = final_state.get("model_params")
                    
                    total_time = time.time() - start_time
                    minutes = int(total_time // 60)
                    seconds = int(total_time % 60)
                    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

                    if model_path and model_path != "No current model exists.":
                        if not metrics:
                            status.update("[bold cyan]Generating final evaluation metrics...[/bold cyan]")
                            try:
                                tool_name = "eval_classification" if "_c" in model_path.lower() else "eval_regression"
                                eval_res = await session.call_tool(tool_name, arguments={"params": {"model_path": model_path}})
                                metrics = eval_res.content[0].text
                            except Exception:
                                pass

                        content = f"[bold green]Model successfully saved at:[/bold green]\n{model_path}\n"
                        
                        if params:
                            content += f"\n[bold cyan]Hyperparameters:[/bold cyan]\n[dim]{params}[/dim]\n"
                            
                        if metrics:
                            try:
                                m_dict = json.loads(metrics)
                                if isinstance(m_dict, dict):
                                    metrics_str = ""
                                    for k, v in m_dict.items():
                                        if k != "classification_report":
                                            val_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                                            metrics_str += f"• {k.replace('_', ' ').title()}: [bold yellow]{val_str}[/bold yellow]  "
                                    content += f"\n[bold magenta]Evaluation Metrics:[/bold magenta]\n{metrics_str}\n"
                                else:
                                    content += f"\n[bold magenta]Evaluation Metrics:[/bold magenta]\n{metrics}\n"
                            except:
                                content += f"\n[bold magenta]Evaluation Metrics:[/bold magenta]\n{metrics}\n"

                        content += f"\n[bold blue]Total Execution Time:[/bold blue] {time_str}"

                        model_panel = Panel(
                            content,
                            title="[bold cyan]Final Output Card[/bold cyan]",
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

    except KeyboardInterrupt:
        console.print("\n[bold red]Pipeline interrupted by user.[/bold red]")
        sys.exit(0)
    except Exception as e:
        console.print_exception(show_locals=False)
        console.print(f"\n[bold red]Pipeline crashed with error:[/bold red] {e}")
        sys.exit(1)

def main():
    try:
        os.system("cls" if os.name == "nt" else "clear")
        asyncio.run(run_pipeline())
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    main()