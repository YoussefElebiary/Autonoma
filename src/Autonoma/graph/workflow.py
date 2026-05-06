from langgraph.graph import StateGraph, END
from Autonoma.graph.state import AutonomaState

from Autonoma.agents.init_node impoty init_agent_node
from Autonoma.agents.analysis_node import analysis_agent_node
from Autonoma.agents.preprocessing_node import preprocessing_agent_node
from Autonoma.agents.modeling_node import modeling_agent_node
from Autonoma.agents.critic_node import critic_agent_node
from Autonoma.agents.executor_node import executor_node

def route_from_critic(state: AutonomaState) -> str:
    """Reads the final_decision and routes to the Executor or back to the Agent."""

    decision = state.get("final_decision", "revise").lower()
    current_agent = state.get("current_agent", "unknown")

    if decision == "approve":
        return "execute"
    
    if current_agent == "analysis":
        return "revise_analysis"
    elif current_agent == "preprocessing":
        return "revise_preprocessing"
    elif current_agent == "modeling":
        return "revise_modeling"
    
    return "end"

def route_after_execution(state: AutonomaState) -> str:
    """After a tool runs, figure out who the NEXT agent in the pipeline is."""

    current_agent = state.get("current_agent", "unknown")

    if current_agent == "analysis":
        return "go_to_preprocessing"
    elif current_agent == "preprocessing":
        return "go_to_modeling"
    
    return "end_pipeline"



workflow = StateGraph(AutonomaState)

# Add all nodes to the graph
workflow.add_node("Initalizer", init_agent_node)
workflow.add_node("Analysis", analysis_agent_node)
workflow.add_node("Preprocessing", preprocessing_agent_node)
workflow.add_node("Modeling", modeling_agent_node)
workflow.add_node("Critic", critic_agent_node)
workflow.add_node("Executor", executor_node)

# Set entry point of the graph
workflow.set_entry_point("Initializer")

# Connect the Initlaizer Node to Analysis Node
workflow.add_edge("Initalizer", "Analysis")

# Connect all agents to Critic
workflow.add_edge("Analysis", "Critic")
workflow.add_edge("Preprocessing", "Critic")
workflow.add_edge("Modeling", "Critic")

# Set Critic Router
workflow.add_conditional_edges(
    "Critic",
    route_from_critic,
    {
        "execute": "Executor",
        "revise_analysis": "Analysis",
        "revise_preprocessing": "Preprocessing",
        "revise_modeling": "Modeling",
        "end": END
    }
)

# Set Executor Router
workflow.add_conditional_edges(
    "Executor",
    route_after_execution,
    {
        "go_to_preprocessing": "Preprocessing",
        "go_to_modeling": "Modeling",
        "end_pipeline": END
    }
)

# Compile Graph
app = workflow.compile()