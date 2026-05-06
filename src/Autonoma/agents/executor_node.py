import json
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ..graph.state import AutonomaState

import os
import sys

server_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server.py"))

server_params = StdioServerParameters(
    command=sys.executable,
    args=[server_script]
)

async def executor_node(state: AutonomaState) -> dict:
    current_agent = state.get("current_agent", "unknown")
    print(f"--- EXECUTING TOOLS FOR: {current_agent.upper()} ---")

    sample_output = state.get("sample_output", "[]")

    try:
        match = re.search(r'```(?:json)?\n(.*?)\n```', sample_output, re.DOTALL)
        clean_output = match.group(1).strip() if match else sample_output.strip()
        tool_calls = json.loads(clean_output)
        observations = []
        model_path = ""

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                for call in tool_calls:
                    tool_name = call.get("tool_name")
                    params = call.get("params", {})
                    print(f"-> Running {tool_name} with params: {params}")

                    result = await session.call_tool(tool_name, arguments=params)

                    result_text = result.content[0].text
                    observations.append(f"Result of {tool_name}:\n{result_text}")

                    if current_agent == 'modeling':
                        try:
                            result_data = json.loads(result_text)
                            if "model_path" in result_data:
                                model_path = result_data["model_path"]
                        except json.JSONDecodeError:
                            print(f"-> Warning: Could not parse model_path from: {result_text}")

        formatted_result = "\n\n".join(observations)

        if current_agent == 'analysis':
            return {
                "eda_insights": formatted_result
            }
        elif current_agent == 'preprocessing':
            return {
                "preprocessing_steps": formatted_result
            }
        elif current_agent == 'modeling':
            return {
                "model_path": model_path
            }
        else:
            return {}
    except json.JSONDecodeError:
        print("-> Executor failed to parse JSON.")
        return {}
    except Exception as e:
        print(f"-> MCP Execution Error: {str(e)}")
        error_msg = f"TOOL EXECUTION FAILED with error: {str(e)}"
        
        if current_agent == 'analysis':
            return {"eda_insights": error_msg}
        elif current_agent == 'preprocessing':
            return {"preprocessing_steps": error_msg}
        else:
            return {}