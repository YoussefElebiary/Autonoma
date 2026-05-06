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

def preprocessing_agent_node(state: AutonomaState) -> dict:
    """The node responsible for processing the data and getting it ready for modeling."""

    print("--- Performing Preprocessing ---")
    data_summary = state.get("data_summary", "No data summary provided.")
    eda_insights = state.get("eda_insights", "No EDA insights provided.")
    critic_feedback = state.get("critic_feedback", "None")
    tool_schemas = state.get("tool_schemas", "No tools provided.")

    prompt_template = init_prompt("preprocessing.txt")
    prompt = prompt_template.format(
        data_summary=data_summary,
        eda_insights=eda_insights,
        critic_feedback=critic_feedback,
        tool_schemas=tool_schemas
    )

    response = llm.invoke(prompt)

    return {
        "sample_output": response.content,
        "current_agent": "preprocessing" 
    }