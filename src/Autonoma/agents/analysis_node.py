import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from Autonoma.graph.state import AutonomaState

def init_prompt(path: str) -> str:
    full_path = Path(__file__).parent.parent / "prompts" / path
    with open(full_path, "r") as f:
        return f.read()

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="gemma-3-4b",
    temperature=0.1
)

def analysis_agent_node(state: AutonomaState) -> dict:
    """The node responsible for reading the raw data summary and outputting insights."""

    print("--- Performing EDA ---")
    data_summary = state.get("data_summary", "No data summary provided.")
    critic_feedback = state.get("critic_feedback", "None")
    tool_schemas = state.get("tool_schemas", "No tools provided.")

    prompt_template = init_prompt("analysis.txt")
    prompt = prompt_template.format(
        data_summary=data_summary,
        critic_feedback=critic_feedback,
        tool_schemas=tool_schemas
    )

    response = llm.invoke(prompt)

    return {
        "draft_output": response.content,
        "current_agent": "analysis" 
    }