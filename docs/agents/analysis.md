# Analysis Agent

The **Analysis Agent** is the first intelligent step in the pipeline. Its role is to perform Exploratory Data Analysis (EDA) and identify potential issues or opportunities in the dataset.

## 🎯 Objectives
- Understand the schema and data types.
- Detect null values, outliers, and skewness.
- Identify correlations and low-variance columns.
- Provide a summary of the data quality to inform the preprocessing phase.

## 🧠 Reasoning Logic
The agent receives:
1. **Data Summary**: A text representation of the dataset (columns, types, etc.).
2. **Tool Schemas**: A list of available analysis tools (e.g., `get_numerical_summary`, `detect_outliers`).
3. **Critic Feedback**: If this is a revision loop, it receives specific instructions on what to fix.

It outputs a **Plan** (in the form of proposed tool calls) which is then sent to the Critic for approval.

## 🛠️ Tools Used
- `get_df_info`: Basic structure of the dataframe.
- `get_numerical_summary`: Statistics for numerical columns.
- `get_categorical_distribution`: Counts and frequencies for categories.
- `detect_outliers`: Identification of data points outside normal ranges.
- `get_correlation`: Matrix of relationships between features.
- `get_skew`: Skewness of numerical columns.
- `get_target_correlation`: Correlation between numerical columns and the target variable.
- `get_column_cardinality`: Number of unique values in a column.
- `check_low_variance`: Check for low-variance columns.

## 📝 Implementation
The node is implemented in `src/autonoma/agents/analysis_node.py` and uses the prompt found in `src/autonoma/prompts/analysis.txt`.

---
[Next: Preprocessing Agent](./preprocessing.md)
