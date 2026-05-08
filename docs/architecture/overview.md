# Architecture Overview

Autonoma is built on a modular, agentic architecture. It separates the **reasoning** (agents) from the **execution** (tools) and the **orchestration** (workflow).

## 🏗️ Core Components

### 1. LangGraph Orchestrator
The backbone of the system is a **LangGraph** state machine. It manages the flow of information between agents and ensures that the pipeline follows a logical sequence while allowing for iterative improvements.

### 2. Specialized Agents
Each agent is a specialized LLM prompt designed for a specific phase of the ML pipeline. They don't execute code directly; instead, they generate "plans" or "tool calls" that are then processed by the Executor.

### 3. MCP Tool Server
The **Model Context Protocol (MCP)** server acts as the execution layer. It exposes a set of Python functions (tools) that agents can call. This decoupling allows the tools to be tested independently and even run in different environments.

### 4. AutonomaState
A centralized state object that travels through the graph. it contains:
- The path to the dataset.
- The current stage of the pipeline.
- Feedback from the Critic.
- Metrics and model paths.

## 🔄 The Agentic Loop

The system follows a repeating pattern for each phase (Analysis, Preprocessing, Modeling):

1. **Plan**: An agent analyzes the current state and decides which tools to use.
2. **Review**: The **Critic Agent** reviews the plan. If it's flawed, the agent must revise it.
3. **Execute**: Once approved, the **Executor** calls the tools via the MCP server.
4. **Reflect**: The results are fed back into the state, and the cycle continues to the next phase.

---

[Next: Workflow Details](./workflow.md)
