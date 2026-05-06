from typing import TypedDict, Optional, Any

class AutonomaState(TypedDict):
    # Dataset Path
    csv_path: str

    # Summary of the DataFrame info returned by the MCP
    data_summary: str
    # Schemas for the available MCP tools
    tool_schemas: str
    # MCP Session object
    mcp_session: Any
    # EDA Insights returned by the Analysis Agent
    eda_insights: str
    # Preprocessing steps approved by critic
    preprocessing_steps: str
    # Path to the current model
    model_path: Optional[str]

    # Currently executing agent
    current_agent: str
    # Sample agent output requiring approve
    sample_output: str
    # Critic feedback based on other agents responses
    critic_feedback: str
    # Limit to critic loop feedbacks to prevent infinite loops
    critic_iterations: int

    # State transfer decision
    final_decision: str