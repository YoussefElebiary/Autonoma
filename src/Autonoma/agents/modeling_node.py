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

def modeling_agent_node(state: AutonomaState) -> dict:
    """The node responsible for using the preprocessed data for making machine learning models."""

    print("--- Performing Modeling ---")
    data_summary = state.get("data_summary", "No data summary provided.")
    eda_insights = state.get("eda_insights", "No EDA insights provided.")
    preprocessing_steps = state.get("preprocessing_steps", "No preprocessing was done.")
    critic_feedback = state.get("critic_feedback", "None")
    tool_schemas = state.get("tool_schemas", "No tools provided.")
    current_model = state.get("model_path", "No current model exists.")

    prompt_template = init_prompt("modeling.txt")
    prompt = prompt_template.format(
        data_summary=data_summary,
        eda_insights=eda_insights,
        preprocessing_steps=preprocessing_steps,
        critic_feedback=critic_feedback,
        tool_schemas=tool_schemas,
        current_model=current_model
    )

    response = llm.invoke(prompt)

    return {
        "sample_output": response.content,
        "current_agent": "modeling" 
    }