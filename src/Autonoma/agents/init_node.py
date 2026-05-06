import json
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

async def init_agent_node(state: AutonomaState) -> dict:
    print("--- PIPELINE START: INITIALIZING DATA ---")
    
    csv_path = state.get("csv_path", "data.csv") 
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                print(f"-> Loading CSV: {csv_path}")
                await session.call_tool("init_state", arguments={"file_path": csv_path})

                print("-> Fetching Data Summary...")
                info_result = await session.call_tool("get_df_info", arguments={})
                
                data_summary = info_result.content[0].text

                print("-> Fetching Tool Schemas...")
                tools_list = await session.list_tools()
                
                schemas = []
                for t in tools_list.tools:
                    schemas.append({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema
                    })
                tool_schemas = json.dumps(schemas, indent=2)

                return {
                    "data_summary": data_summary,
                    "tool_schemas": tool_schemas,
                    "critic_iterations": 0,
                    "critic_feedback": "None"
                }
                
    except Exception as e:
        print(f"-> Initialization Error: {e}")
        return {
            "data_summary": f"FAILED TO LOAD DATA: {str(e)}",
            "tool_schemas": "FAILED TO LOAD SCHEMAS",
            "critic_iterations": 0,
            "critic_feedback": "None"
        }