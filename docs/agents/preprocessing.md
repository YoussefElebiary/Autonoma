# Preprocessing Agent

The **Preprocessing Agent** takes the raw data and the insights from the Analysis phase and transforms them into a format suitable for machine learning.

## 🎯 Objectives
- Clean the data (handle missing values, drop duplicates).
- Transform features (scaling, encoding, normalization).
- Perform feature engineering (creating new columns based on existing ones).
- Split the data into training, validation, and test sets.

## 🧠 Reasoning Logic
The agent utilizes:
1. **Data Summary**: To check current columns and types.
2. **EDA Insights**: To remember which columns had outliers, nulls, or high correlation.
3. **Tool Schemas**: Available tools like `drop_column`, `fill_nulls`, `scale_column`, and `split_data`.
4. **Critic Feedback**: To refine the cleaning strategy if previously rejected.

It generates a sequence of tool calls that represent the "Cleaning Pipeline".

## 🛠️ Key Tools
- `fill_nulls`: Imputes missing data using mean, median, or constant values.
- `transform_column`: Applies mathematical transformations (log, sqrt).
- `encode_categorical`: Converts text labels into numerical formats (One-Hot, Label).
- `split_data`: Mandatory step to create `X_train`, `y_train`, etc.
- `scale_column`: Normalizes numerical ranges (StandardScaler, MinMaxScaler).

## 📝 Implementation
Found in `src/autonoma/agents/preprocessing_node.py`, using `src/autonoma/prompts/preprocessing.txt`.

---
[Next: Modeling Agent](./modeling.md)
