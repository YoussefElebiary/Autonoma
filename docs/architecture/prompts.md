# Prompt Engineering

Autonoma relies on high-quality prompts to guide the agents through the complex steps of the machine learning pipeline.

## 📂 Prompt Storage
All system prompts are stored as plain `.txt` files in the `src/autonoma/prompts/` directory. This makes them easy to version control and modify without touching the Python code.

### Available Prompts
- `analysis.txt`: Guides the agent to perform EDA and identify data issues.
- `preprocessing.txt`: Focuses on data cleaning, transformation, and splitting.
- `modeling.txt`: Instructions for model selection, tuning, and evaluation.
- `critic.txt`: The "Judge" prompt that evaluates plans for logical and statistical correctness.

## 🧩 Dynamic Injection
Prompts are not static; they are templates that are filled with real-time data from the `AutonomaState`.

### Common Injection Variables
- `{data_summary}`: Injected into almost every prompt so the agent "sees" the data structure.
- `{tool_schemas}`: The JSON schemas of the tools available for the current phase.
- `{critic_feedback}`: If the Critic has requested a revision, the feedback is injected here to guide the agent's next attempt.
- `{eda_insights}`: Used in Preprocessing and Modeling to provide context from earlier phases.

## 📝 Example Template (Simplified)
```text
You are an expert Data Scientist. 
Current Data Summary: {data_summary}
Previous Critic Feedback: {critic_feedback}

Available Tools:
{tool_schemas}

Your task is to propose a list of tool calls to clean this data.
```

## 🔧 Fine-Tuning Prompts
If you find that an agent is consistently making the same mistake (e.g., forgetting to scale a specific column), you can update the corresponding `.txt` file to add specific constraints or examples.

---
[Back to Index](../index.md)
