# Autonoma: Autonomous ML Pipeline Orchestrator

**Autonoma** is a state-of-the-art, AI-driven machine learning pipeline designed to automate the end-to-end data science workflow. Leveraging **LangGraph** for orchestration and the **Model Context Protocol (MCP)** for modular tool execution, Autonoma transforms raw datasets into optimized, production-ready models with minimal human intervention.

---

## 🚀 Key Features

- **Agentic Orchestration**: Uses a `Planner -> Executor -> Critic` loop to ensure high-quality preprocessing and modeling decisions.
- **MCP Tooling Architecture**: Modularized tools for EDA, data cleaning, feature engineering, and model training, decoupled via a standalone MCP server.
- **Dynamic Decision Making**: An AI Critic evaluates every step, requesting revisions if preprocessing or modeling logic doesn't meet quality standards.
- **Comprehensive ML Support**: Built-in support for Linear Models, Tree-based models (Random Forest, Decision Trees), and Gradient Boosting (XGBoost).
- **Professional CLI**: A high-fidelity command-line interface built with `rich`, providing real-time status updates, beautiful panels, and detailed metrics.

---

## 🛠️ Project Structure

```text
Autonoma/
├── src/autonoma/
│   ├── agents/          # AI Agent definitions (Analysis, Preprocessing, Modeling, Critic)
│   ├── graph/           # LangGraph workflow and state management
│   ├── tools/           # Core logic for EDA, Preprocessing, and Modeling
│   ├── main.py          # CLI Entry point
│   ├── server.py        # MCP Server (FastMCP)
│   └── config.py        # Project configurations and LLM settings
├── models/              # Directory where trained .joblib models are saved
├── tests/               # Unit and integration tests
├── docs/                # Detailed documentation
├── Pipfile              # Dependency management
└── README.md            # You are here
```

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.13+**
- **Pipenv** (`pip install pipenv`)

### Step 1: Install Dependencies
Clone the repository and install the required packages using Pipenv:

```bash
git clone https://github.com/YoussefElebiary/Autonoma.git
cd Autonoma
pipenv install
```

> [!NOTE]
> The core dependencies include `langgraph`, `mcp`, `polars`, `scikit-learn`, `xgboost`, `rich`, and `python-dotenv`.

### Step 2: Configure Environment
Create a `.env` file in the root directory. Autonoma is designed to work with any **OpenAI-compatible SDK** and defaults to **LM Studio** for local inference.

```bash
# Optional: If using LM Studio, the key can be anything or "lm-studio"
LLM_API_KEY=lm-studio
# Optional: Customize the model or base URL in src/autonoma/config.py
```

> [!TIP]
> You can easily switch to other providers like OpenAI, Anthropic, or Groq by updating the `LLM_BASE_URL` and `LLM_MODEL` in `src/autonoma/config.py`.


---

## 🏃 Running Autonoma

To start the autonomous pipeline, run the following command:

```bash
pipenv run python src/autonoma/main.py
```

### The Workflow:
1. **Input**: Provide the path to your CSV dataset when prompted.
2. **Analysis**: The pipeline performs automated EDA (cardinality, null checks, correlations).
3. **Preprocessing**: Agents apply transformations, handle nulls, and encode features.
4. **Modeling**: The system performs hyperparameter tuning (Grid/Random Search) and selects the best model.
5. **Critique**: A Critic agent reviews the results and may trigger a "Revise" loop for better accuracy.
6. **Output**: The final model is saved to the `models/` directory, and a detailed performance card is displayed.

---

## 🚢 Deployment Considerations

If you are planning to deploy Autonoma to a production or cloud environment, consider the following:

### 1. Environment Variables
Ensure the following variables are set in your deployment environment:
- `LLM_API_KEY`: Required for agent reasoning.
- `VERBOSE_MODE`: Set to `true` for detailed logging.

### 2. Model Persistence
Autonoma saves models locally in the `models/` directory by default. For cloud deployments, you may want to:
- Map a persistent volume to the `/models` directory.
- Update `src/autonoma/tools/modeling.py` to upload models to S3 or a similar blob storage.

### 3. MCP Server Scaling
The MCP server (`server.py`) currently runs as a local subprocess. In a distributed architecture, you can host the MCP server as a standalone service (e.g., via Docker) and connect the `main.py` client to it over a network transport.

### 4. Resource Allocation
- **CPU/RAM**: Since it uses `Polars` and `Scikit-learn`, ensure sufficient memory for large datasets.
- **LLM Usage**: The pipeline is multi-turn; monitor your API usage as complex datasets may require multiple iterations between the agent and the critic.

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by Youssef Elebiary*