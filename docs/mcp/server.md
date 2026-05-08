# MCP Server (FastMCP)

Autonoma uses a **Model Context Protocol (MCP)** server to encapsulate its data science tools. This architecture provides a clean separation between the "brain" (LLM agents) and the "tools" (Python logic).

## 🚀 Implementation
The server is implemented in `src/autonoma/server.py` using the **FastMCP** framework.

### Why MCP?
- **Standardization**: MCP provides a universal way for LLMs to interact with external tools.
- **Modularity**: Tools can be added, removed, or updated without changing the core agent logic.
- **Security**: The server runs in a separate process, providing a layer of isolation.

## 💾 State Management (Server-Side)
The MCP server maintains an internal `STATE` dictionary to hold the Polars DataFrames in memory:
- `df`: The current working dataset.
- `X_train`, `y_train`, etc.: The split datasets.
- `model_path`: Path to the last saved model.

This allows the agent to call multiple tools in sequence without needing to pass the entire dataset back and forth across the protocol.

## 🛠️ Exposing Tools
Tools are exposed using the `@mcp.tool()` decorator. Each tool uses **Pydantic** schemas (defined in `src/autonoma/schema/tools_io.py`) to validate input arguments.

### Example Tool Definition
```python
@mcp.tool()
def fill_nulls(params: FillNullsSchema) -> str:
    # Logic to fill nulls using PreprocessingTools
    return "Success"
```

## 📡 Communication
The server communicates via **Standard I/O (stdio)**, which is the default transport for MCP. The `main.py` script starts the server as a subprocess and establishes a `ClientSession` to interact with it.

---
[Back to Index](../index.md)
