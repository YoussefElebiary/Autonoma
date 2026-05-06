import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from ..graph.state import AutonomaState
from ..config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, debug_print

def init_prompt(path: str) -> str:
    full_path = Path(__file__).parent.parent / "prompts" / path
    with open(full_path, "r") as f:
        return f.read()


llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE
)

def modeling_agent_node(state: AutonomaState) -> dict:
    """The node responsible for using the preprocessed data for making machine learning models."""

    debug_print("--- Performing Modeling ---")
    data_summary = state.get("data_summary", "No data summary provided.")
    eda_insights = state.get("eda_insights", "No EDA insights provided.")
    preprocessing_steps = state.get("preprocessing_steps", "No preprocessing was done.")
    critic_feedback = state.get("critic_feedback", "None")
    try:
        schemas_dict = json.loads(state.get("tool_schemas", "{}"))
        tool_schemas = json.dumps(schemas_dict.get("modeling", []), indent=2)
    except json.JSONDecodeError:
        tool_schemas = "No tools provided."
    model_path = state.get("model_path", "No current model exists.")

    prompt_template = init_prompt("modeling.txt")
    prompt = prompt_template.format(
        data_summary=data_summary,
        eda_insights=eda_insights,
        preprocessing_steps=preprocessing_steps,
        critic_feedback=critic_feedback,
        tool_schemas=tool_schemas,
        model_path=model_path
    )

    response = llm.invoke(prompt)

    return {
        "sample_output": response.content,
        "current_agent": "modeling" 
    }