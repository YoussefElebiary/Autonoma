import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from Autonoma.graph.state import AutonomaState

server_params = StdioServerParameters(
    command="python",
    args=["-m", "Autonoma.server"]
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

                return {
                    "data_summary": data_summary,
                    "critic_iterations": 0,
                    "critic_feedback": "None"
                }
                
    except Exception as e:
        print(f"-> Initialization Error: {e}")
        return {
            "data_summary": f"FAILED TO LOAD DATA: {str(e)}",
            "critic_iterations": 0,
            "critic_feedback": "None"
        }