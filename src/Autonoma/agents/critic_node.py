import json
import re
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

def critic_agent_node(state: AutonomaState) -> dict:
    """The node responsible for providing feedback and routing decisions."""

    current_agent = state.get("current_agent", "unknown")
    print(f"--- CRITIQUE: Reviewing {current_agent.upper()} Agent ---")

    critic_iterations = state.get("critic_iterations", 0)
    if critic_iterations >= 3:
        print("-> Max critic iterations reached. Forcing approval.")
        return {
            "final_decision": "approve",
            "critic_iterations": 0
        }

    data_summary = state.get("data_summary", "No data summary provided.")
    eda_insights = state.get("eda_insights", "No EDA insights provided.")
    preprocessing_steps = state.get("preprocessing_steps", "No preprocessing was done.")
    current_model = state.get("model_path", "No current model exists.")
    
    sample_output = state.get("sample_output", "[]")

    prompt_template = init_prompt("critic.txt")
    prompt = prompt_template.format(
        current_agent=current_agent,
        data_summary=data_summary,
        eda_insights=eda_insights,
        preprocessing_steps=preprocessing_steps,
        current_model=current_model,
        sample_output=sample_output
    )

    response = llm.invoke(prompt)
    try:
        content = response.content
        match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
        clean_content = match.group(1).strip() if match else content.strip()
        critic_decision = json.loads(clean_content)
        feedback = critic_decision.get("feedback", "Looks good.")
        decision = critic_decision.get("decision", "approve").lower()

        new_iters = 0 if decision == 'approve' else critic_iterations + 1
        
        print(f"-> Decision: {decision.upper()} (Iteration {new_iters}/3)")

        return {
            "critic_feedback": feedback,
            "final_decision": decision,
            "critic_iterations": new_iters
        }
    except json.JSONDecodeError:
        print("-> Critic output invalid JSON. Forcing approval to unblock pipeline.")
        return {
            "critic_feedback": "None",
            "final_decision": "approve",
            "critic_iterations": 0
        }