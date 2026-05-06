import json
from ..graph.state import AutonomaState

async def init_agent_node(state: AutonomaState) -> dict:
    print("--- PIPELINE START: INITIALIZING DATA ---")
    
    csv_path = state.get("csv_path", "data.csv") 
    session = state.get("mcp_session")
    
    try:
        print(f"-> Loading CSV: {csv_path}")
        init_res = await session.call_tool("init_state", arguments={"params": {"file_path": csv_path}})
        init_text = init_res.content[0].text
        print(f"-> Init Result: {init_text}")
        if "Error" in init_text or "Field required" in init_text:
            raise Exception(f"Failed to initialize state: {init_text}")

        print("-> Fetching Data Summary...")
        info_result = await session.call_tool("get_df_info", arguments={})
        
        data_summary = info_result.content[0].text

        print("-> Fetching Tool Schemas...")
        tools_list = await session.list_tools()
        
        schemas = []
        for t in tools_list.tools:
            if t.name == "init_state":
                continue
            
            defs = t.inputSchema.get("$defs", {})
            schema_keys = list(defs.keys())
            if schema_keys:
                actual_schema = defs[schema_keys[0]]
                props = actual_schema.get("properties", {})
            else:
                props = t.inputSchema.get("properties", {})
                
            simple_schema = {k: v.get("type", "any") for k, v in props.items()}
            schemas.append({
                "name": t.name,
                "description": t.description,
                "inputSchema": simple_schema
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