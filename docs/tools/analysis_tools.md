# Analysis Tools

The Analysis tools are used by the Analysis Agent to understand the dataset's characteristics and identify potential data quality issues.

## 🛠️ Tool List

### `get_df_info`
Returns the basic structure of the dataframe.
- **Output**: Shape, column names, data types, and null counts for every column.

### `get_numerical_summary`
Calculates descriptive statistics for numerical columns.
- **Parameters**: `columns` (optional list of column names).
- **Output**: Mean, std, min, max, and quartiles for each column.

### `get_categorical_distribution`
Analyzes the frequency of values in a categorical column.
- **Parameters**: `column` (name), `top_n` (number of top categories to return).
- **Output**: Counts and percentages of the most frequent values.

### `detect_outliers`
Identifies data points that deviate significantly from the mean.
- **Parameters**: `column` (name), `threshold` (Z-score threshold, default: 3.0).
- **Output**: Count and percentage of outliers in the column.

### `get_correlation`
Generates a Pearson correlation matrix for all numerical columns.
- **Output**: Matrix of correlation values and the list of columns included.

### `get_skew`
Calculates the skewness of numerical features.
- **Output**: Skewness values for each column. High values suggest the need for transformation (e.g., log).

### `get_target_correlation`
Calculates the correlation of all numerical features with a specific target variable.
- **Parameters**: `target` (name of the target column).
- **Output**: List of correlation coefficients relative to the target.

### `get_column_cardinality`
Checks the number of unique values in a categorical column.
- **Parameters**: `column` (name).
- **Output**: `n_unique` count. Useful for identifying high-cardinality features that might need special encoding.

### `check_low_variance`
Identifies columns where a single value dominates, making them potentially uninformative.
- **Parameters**: `threshold` (percentage of the most frequent value, default: 0.95).
- **Output**: List of "quasi-constant" columns and their frequencies.

---
[Next: Preprocessing Tools](./preprocessing_tools.md)
