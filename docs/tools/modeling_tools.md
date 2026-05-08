# Modeling Tools

The Modeling tools facilitate model training, hyperparameter optimization, and performance evaluation.

## 🛠️ Tool List

### `linear_models`
Trains linear or logistic regression models.
- **Algorithms**: `linear`, `logistic`, `lasso`, `ridge`, `elastic`.
- **Saving**: Automatically saves the trained model as a `.joblib` file in the `models/` directory.

### `tree_models`
Trains tree-based or ensemble models.
- **Algorithms**: `decision_c` (classifier), `decision_r` (regressor), `forest_c`, `forest_r`, `xgb_c`, `xgb_r`.
- **Early Stopping**: XGBoost models automatically use the validation set for early stopping to prevent overfitting.

### `grid_search`
Performs an exhaustive search over a specified parameter grid.
- **Cross-Validation**: Defaults to 5-fold CV.
- **Output**: Returns the best parameters and saves the best estimator.

### `random_search`
Performs a randomized search over parameter distributions.
- **Efficiency**: More efficient than Grid Search for large search spaces.
- **Parameters**: `iters` controls the number of parameter settings sampled.

### `eval_classification`
Comprehensive evaluation for classification tasks.
- **Metrics**: Accuracy, Recall, Precision, F1-Score (macro), ROC-AUC.
- **Report**: Includes a full classification report (per-class metrics).

### `eval_regression`
Comprehensive evaluation for regression tasks.
- **Metrics**: MSE (Mean Squared Error), RMSE (Root Mean Squared Error), MAE (Mean Absolute Error).

## 📁 Model Storage
All models are saved using `joblib` with a timestamped filename:
`[model_type]_[YYYYMMDD_HHMM].joblib`

The absolute path to the most recently trained model is always updated in the `AutonomaState`.
