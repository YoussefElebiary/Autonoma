# Autonoma Documentation

Welcome to the official documentation for **Autonoma**, an autonomous machine learning pipeline. 

Autonoma is designed to handle the complexity of data science workflows by using an agentic approach. Instead of a rigid script, it employs a collection of specialized AI agents that collaborate to analyze, preprocess, and model data.

### 🧠 Core Philosophy: Brain Switching
Autonoma uses a "Brain Switching" architecture managed by LangGraph. This eliminates the need for massive context windows by passing only distilled state between agents, making it possible to run production-grade ML pipelines on **consumer hardware with limited VRAM**.

## 📚 Documentation Sections

### [Architecture](./architecture/overview.md)
Understand the high-level design, the LangGraph-based workflow, and how state is managed across the pipeline.

### [Agents](./agents/analysis.md)
Detailed breakdown of each agent in the system:
- **Initializer**: Sets up the state and loads the dataset.
- **Analysis Agent**: Performs Exploratory Data Analysis (EDA).
- **Preprocessing Agent**: Handles data cleaning and feature engineering.
- **Modeling Agent**: Trains models and performs hyperparameter tuning.
- **Critic Agent**: Reviews decisions and ensures quality.
- **Executor**: The bridge between agents and the MCP tools.

### [Tools](./tools/analysis_tools.md)
Documentation for the underlying tools that perform the heavy lifting:
- **Analysis Tools**: Statistical analysis and distribution checks.
- **Preprocessing Tools**: Transformations, null handling, and encoding.
- **Modeling Tools**: Training, evaluation, and tuning.

### [MCP Server](./mcp/server.md)
Learn about the Model Context Protocol (MCP) server that powers the tool execution layer.

---

## 🛠️ Getting Started
If you haven't installed Autonoma yet, please refer to the [main README](../README.md) for installation and quick-start instructions.
