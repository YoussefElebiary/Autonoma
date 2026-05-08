# State Management

The `AutonomaState` is the single source of truth for the entire pipeline. It is a `TypedDict` that is passed between nodes in the LangGraph workflow.

## 📋 State Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `csv_path` | `str` | The absolute path to the dataset being processed. |
| `data_summary` | `str` | A string summary of the dataset (columns, types, nulls). |
| `tool_schemas` | `str` | JSON-formatted schemas for the tools available to the current agent. |
| `mcp_session` | `Any` | The active MCP client session for calling tools. |
| `eda_insights` | `str` | Findings from the Analysis phase. |
| `preprocessing_steps`| `str` | A list of approved transformations to be applied. |
| `model_path` | `Optional[str]` | The file path to the best-performing model. |
| `current_agent` | `str` | The identifier of the agent currently in control. |
| `critic_feedback` | `str` | Instructions from the Critic for revising the current plan. |
| `critic_iterations` | `int` | Counter for how many times the Critic has intervened. |
| `modeling_iterations`| `int` | Counter for modeling attempts. |
| `evaluation_metrics` | `Optional[str]`| Final model performance metrics (JSON string). |
| `final_decision` | `str` | The routing decision (`approve`, `revise`, etc.). |

## 🛠️ How State is Updated

Each node in the graph returns a partial update to the state. LangGraph then merges these updates into the main state object.

### Example: Analysis Node Update
```python
{
    "current_agent": "analysis",
    "sample_output": agent_response,
    "eda_insights": analysis_results
}
```

### Example: Critic Node Update
```python
{
    "final_decision": "approve",
    "critic_feedback": "The proposed preprocessing steps are statistically sound."
}
```

## 🧊 Persistence
While the state is currently transient (held in memory during a run), the design allows for easy integration with LangGraph's checkpointers to allow for "pausable" and "resumable" pipelines in the future.
