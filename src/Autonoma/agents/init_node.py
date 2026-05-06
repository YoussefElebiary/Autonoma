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
        
        analysis_tools = ["get_df_info", "get_numerical_summary", "get_categorical_distribution", "detect_outliers", "get_correlation", "get_skew", "get_target_correlations", "get_column_cardinality", "check_low_variance"]
        preprocessing_tools = ["drop_column", "fill_nulls", "drop_outliers", "transform_column", "scale_column", "encode_categorical", "split_data", "create_feature"]
        modeling_tools = ["grid_search", "random_search", "linear_models", "tree_models", "eval_classification", "eval_regression"]

        schemas_dict = {
            "analysis": [],
            "preprocessing": [],
            "modeling": []
        }
        
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
                
            simple_schema = {}
            for k, v in props.items():
                if "enum" in v:
                    simple_schema[k] = f"enum: {v['enum']}"
                else:
                    simple_schema[k] = v.get("type", "any")
                    
            tool_dict = {
                "name": t.name,
                "description": t.description,
                "inputSchema": simple_schema
            }

            if t.name in analysis_tools:
                schemas_dict["analysis"].append(tool_dict)
            elif t.name in preprocessing_tools:
                schemas_dict["preprocessing"].append(tool_dict)
            elif t.name in modeling_tools:
                schemas_dict["modeling"].append(tool_dict)
                
        tool_schemas = json.dumps(schemas_dict)

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