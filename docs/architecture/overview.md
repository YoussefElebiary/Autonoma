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
- **`final_decision`**: The control signal that determines the graph's path.

## 🧠 "Brain Switching" Architecture

A core design principle of Autonoma is **Brain Switching**. Instead of using a single LLM session with a massive, monolithic context window, the system uses LangGraph to orchestrate specialized, stateless turns.

### Why this matters:
1. **Consumer Hardware Support**: By passing only the distilled "insights" from one agent to the next, the context size remains small. This allows Autonoma to run high-quality pipelines using local models (via LM Studio) on hardware with limited VRAM.
2. **Context Compression**: The system treats the `AutonomaState` as a living summary. The LLM never sees the raw multi-turn conversation; it only sees the current data schema and the summarized results of previous stages.
3. **Task Specialization**: Each "Brain" (Agent) is highly focused on its specific domain (Analysis, Modeling, etc.), reducing the risk of "instruction drift" often seen in long chat sessions.

The system follows a repeating pattern for each phase (Analysis, Preprocessing, Modeling):

1. **Plan**: An agent analyzes the current state and decides which tools to use.
2. **Review**: The **Critic Agent** reviews the plan. If it's flawed, the agent must revise it.
3. **Execute**: Once approved, the **Executor** calls the tools via the MCP server.
4. **Reflect**: The results are fed back into the state, and the cycle continues to the next phase.

## 🧠 The Role of the LLM

The "intelligence" of Autonoma is externalized to the LLM. The performance of the pipeline (how well it cleans data and which models it selects) is a direct reflection of the LLM's reasoning capabilities.

- **Reasoning**: Higher-tier models (GPT-4o, Claude 3.5) are better at understanding complex data distributions and identifying subtle preprocessing needs.
- **Accuracy**: The Critic agent requires strong logic to catch errors in the tool calls proposed by other agents.
- **Context Efficiency**: By using LangGraph state to manage data, the LLM does not need to maintain a massive conversation history. Each agent receives only the **distilled state** relevant to its task, enabling "Brain Switching" that works efficiently on consumer hardware with limited VRAM.

---

[Next: Workflow Details](./workflow.md)
