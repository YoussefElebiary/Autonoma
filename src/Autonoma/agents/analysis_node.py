import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from ..graph.state import AutonomaState

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
    try:
        schemas_dict = json.loads(state.get("tool_schemas", "{}"))
        tool_schemas = json.dumps(schemas_dict.get("analysis", []), indent=2)
    except json.JSONDecodeError:
        tool_schemas = "No tools provided."

    prompt_template = init_prompt("analysis.txt")
    prompt = prompt_template.format(
        data_summary=data_summary,
        critic_feedback=critic_feedback,
        tool_schemas=tool_schemas
    )

    response = llm.invoke(prompt)

    return {
        "sample_output": response.content,
        "current_agent": "analysis" 
    }