# Modeling Agent

The **Modeling Agent** is responsible for selecting, training, and optimizing the machine learning models.

## 🎯 Objectives
- Determine if the task is **Classification** or **Regression**.
- Select appropriate algorithms (Linear, Trees, Ensembles).
- Define hyperparameter search spaces.
- Evaluate model performance and iterate if necessary.

## 🧠 Reasoning Logic
The agent consumes:
1. **Full Context**: Data summary, EDA insights, and the exact preprocessing steps taken.
2. **Modeling History**: Results from previous modeling attempts and the current iteration count.
3. **Tool Schemas**: Tools for training (`tree_models`, `linear_models`) and tuning (`grid_search`, `random_search`).
4. **Critic Feedback**: Guidance on model selection or metric improvement.

### 🔄 Retry & Self-Enhancement
The Modeling Agent is unique because it operates in a multi-turn loop where it consumes its own performance data.

#### 1. Metric Capture
When the Agent calls `eval_classification` or `eval_regression`, the **Executor** captures the JSON result. It extracts key metrics (like `accuracy`, `f1`, or `rmse`) and stores them in the `modeling_results` field of the state.

#### 2. Incorporation
In the next turn, these results are injected into the Agent's prompt under the `PREVIOUS MODELING RESULTS` section. 

#### 3. Enhancement Loop
The Agent observes these metrics and compares them against the goal. If the metrics are poor (e.g., Accuracy is 0.5), the Agent uses its reasoning capabilities to:
- **Analyze failure**: "The Random Forest is overfitting."
- **Adjust strategy**: "I will increase regularization or try XGBoost with a different learning rate."
- **Propose improvements**: The Agent outputs a new set of tool calls to train a better model.

This loop continues until the Agent is satisfied with the metrics (signaled by returning `[]`) or the `MAX_MODELING_ITERATIONS` limit is reached.

## 🛠️ Key Tools
- `linear_models`: Logistic Regression, Lasso, Ridge, ElasticNet.
- `tree_models`: Decision Trees, Random Forest, XGBoost.
- `grid_search`: Exhaustive search over specified parameter values.
- `random_search`: Randomized search over parameter distributions.
- `eval_classification` / `eval_regression`: Comprehensive metrics (F1, Accuracy, MSE, etc.).

## 📝 Implementation
Found in `src/autonoma/agents/modeling_node.py`, using `src/autonoma/prompts/modeling.txt`.

---
[Next: Critic Agent](./critic.md)
