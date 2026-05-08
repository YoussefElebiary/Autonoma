import json
import re
from ..graph.state import AutonomaState
from ..config import debug_print

async def executor_node(state: AutonomaState) -> dict:
    current_agent = state.get("current_agent", "unknown")
    debug_print(f"--- EXECUTING TOOLS FOR: {current_agent.upper()} ---")

    sample_output = state.get("sample_output", "[]")
    session = state.get("mcp_session")

    try:
        match = re.search(r'```(?:json)?\n(.*?)\n```', sample_output, re.DOTALL)
        clean_output = match.group(1).strip() if match else sample_output.strip()
        tool_calls = json.loads(clean_output)
        observations = []
        model_path = state.get("model_path", "")
        model_params = state.get("model_params", None)
        evaluation_metrics = state.get("evaluation_metrics", None)

        for call in tool_calls:
            tool_name = call.get("tool_name")
            params = call.get("params", {})
            debug_print(f"-> Running {tool_name} with params: {params}")

            result = await session.call_tool(tool_name, arguments={"params": params})
            
            result_text = result.content[0].text
            observations.append(f"Result of {tool_name}:\n{result_text}")

            try:
                result_data = json.loads(result_text)
                if current_agent == 'modeling':
                    if "model_path" in result_data:
                        model_path = result_data["model_path"]
                    
                    # Capture best params from search tools
                    if "params" in result_data:
                        model_params = json.dumps(result_data["params"], indent=2)
                    # Or capture direct training params
                    elif "params" in params:
                         model_params = json.dumps(params["params"], indent=2)

                    # Capture metrics from eval tools
                    if any(k in result_data for k in ["accuracy", "mse", "f1", "rmse"]):
                        evaluation_metrics = result_text

            except json.JSONDecodeError:
                if "Error" in result_text or "validation error" in result_text or "failed" in result_text.lower():
                    debug_print(f"-> Tool {tool_name} failed: {result_text}")
                else:
                    debug_print(f"-> Tool {tool_name} returned non-JSON: {result_text[:50]}...")

        formatted_result = "\n\n".join(observations)
        
        # Check if any tool failed to signal a retry
        has_error = "Error" in formatted_result or "failed" in formatted_result.lower() or "validation error" in formatted_result.lower()
        new_iters = state.get("critic_iterations", 0) + (1 if has_error else 0)

        update = {
            "critic_iterations": new_iters,
            "final_decision": "revise" if has_error else "approve"
        }

        if current_agent == 'analysis':
            update["eda_insights"] = formatted_result
        elif current_agent == 'preprocessing':
            update["preprocessing_steps"] = formatted_result
        elif current_agent == 'modeling':
            update.update({
                "model_path": model_path,
                "modeling_results": formatted_result,
                "model_params": model_params,
                "evaluation_metrics": evaluation_metrics
            })
        
        return update

    except json.JSONDecodeError:
        debug_print("-> Executor failed to parse JSON.")
        return {
            "critic_iterations": state.get("critic_iterations", 0) + 1,
            "final_decision": "revise"
        }
    except Exception as e:
        debug_print(f"-> MCP Execution Error: {str(e)}")
        error_msg = f"TOOL EXECUTION FAILED with error: {str(e)}"
        
        update = {
            "critic_iterations": state.get("critic_iterations", 0) + 1,
            "final_decision": "revise"
        }

        if current_agent == 'analysis':
            update["eda_insights"] = error_msg
        elif current_agent == 'preprocessing':
            update["preprocessing_steps"] = error_msg
        
        return update