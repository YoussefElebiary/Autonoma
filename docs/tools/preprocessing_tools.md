# Preprocessing Tools

The Preprocessing tools handle data cleaning, transformation, and preparation. Most of these tools are designed to update both the main dataframe (`df`) and the split datasets (`X_train`, `X_val`, `X_test`) simultaneously to ensure consistency.

## 🛠️ Tool List

### `drop_column`
Removes one or more columns from all datasets in the current state.

### `fill_nulls`
Imputes missing values using various strategies.
- **Methods**: `median`, `mode`, `below` (fill), `knn`, or `custom` value.
- **Consistency**: When using `median`, `mode`, or `knn`, the parameters are fitted on the training set and applied to the validation and test sets.

### `drop_outliers`
Filters out rows based on Z-score thresholds.
- **Threshold**: Number of standard deviations from the mean (default: 3.0).

### `transform_column`
Applies mathematical transformations to numerical features.
- **Methods**: `log`, `exp`, `box-cox`, `yeo-johnson`.
- **Note**: Automatic handling of `log1p` to avoid infinity on zero values.

### `scale_column`
Normalizes the range of numerical features using Scikit-learn scalers.
- **Methods**: `standard` (Z-score), `minmax` (0-1), `maxabs`, `robust` (median/IQR), `quantile`.
- **Strictness**: Scalers are always fitted on `X_train` and applied to `X_val`/`X_test`.

### `encode_categorical`
Converts categorical strings into numerical values.
- **Methods**:
    - `onehot`: One-Hot Encoding (drops the first category to avoid dummy variable trap).
    - `label`: Basic integer encoding.
    - `ordinal`: Maps categories to integers based on observed order.
    - `target`: Encodes categories based on their relationship with the target variable.
    - `custom`: Uses a user-provided dictionary mapping.

### `split_data`
Splits the main dataframe into Train, Validation, and (optional) Test sets.
- **Stratification**: Automatically stratifies based on the target column to maintain class distribution.
- **Note**: This tool initializes the `X_train`, `y_train`, etc. fields in the state.

### `create_feature`
Declarative feature engineering.
- **Operations**:
    - `arithmetic`: Basic math between two columns (+, -, *, /).
    - `date_extract`: Pulls month, day, and weekday from datetime columns.

---
[Next: Modeling Tools](./modeling_tools.md)
