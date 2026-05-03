import pytest
from unittest.mock import patch, ANY
import polars as pl
from Autonoma.server import (
    STATE,
    grid_search,
    random_search,
    linear_models,
    tree_models,
    eval_classification,
    eval_regression
)
from Autonoma.schema.tools_io import (
    GridSearchSchema,
    RandomSearchSchema,
    LinearModelsSchema,
    TreeModelsSchema,
    EvalClassificationSchema,
    EvalRegressionSchema
)

def test_grid_search_no_data():
    params = GridSearchSchema(model_type="logistic", params={"C": [0.1, 1.0]})
    assert grid_search(params) == "Train dataset is not yet split. Use 'split_data' first"

@patch("Autonoma.server.ModellingTools.grid_search")
def test_grid_search_success(mock_tool, mock_df):
    STATE['X_train'] = mock_df
    STATE['y_train'] = mock_df
    mock_tool.return_value = {"params": {"C": 1.0}, "score": 0.9}
    params = GridSearchSchema(model_type="logistic", params={"C": [0.1, 1.0]})
    res = grid_search(params)
    assert res == "{'params': {'C': 1.0}, 'score': 0.9}"
    mock_tool.assert_called_once_with(X_train=ANY, y_train=ANY, model_type="logistic", params={"C": [0.1, 1.0]}, folds=5)

def test_random_search_no_data():
    params = RandomSearchSchema(model_type="forest_c", params={"n_estimators": [10, 50]})
    assert random_search(params) == "Train dataset is not yet split. Use 'split_data' first"

@patch("Autonoma.server.ModellingTools.random_search")
def test_random_search_success(mock_tool, mock_df):
    STATE['X_train'] = mock_df
    STATE['y_train'] = mock_df
    mock_tool.return_value = {"params": {"n_estimators": 50}, "score": 0.85}
    params = RandomSearchSchema(model_type="forest_c", params={"n_estimators": [10, 50]})
    res = random_search(params)
    assert res == "{'params': {'n_estimators': 50}, 'score': 0.85}"
    mock_tool.assert_called_once_with(X_train=ANY, y_train=ANY, model_type="forest_c", params={"n_estimators": [10, 50]}, folds=5, iters=100)

def test_linear_models_no_data():
    params = LinearModelsSchema(model="logistic", params={"C": 1.0})
    assert linear_models(params) == "Train dataset is not yet split. Use 'split_data' first"

@patch("Autonoma.server.ModellingTools.linear_models")
def test_linear_models_success(mock_tool, mock_df):
    STATE['X_train'] = mock_df
    STATE['y_train'] = mock_df
    mock_tool.return_value = {"model_path": "path/to/model"}
    params = LinearModelsSchema(model="logistic", params={"C": 1.0})
    res = linear_models(params)
    assert res == "{'model_path': 'path/to/model'}"
    assert STATE['model_path'] == "path/to/model"
    mock_tool.assert_called_once_with(X_train=ANY, y_train=ANY, model="logistic", params={"C": 1.0}, fit=True)

def test_tree_models_no_data():
    params = TreeModelsSchema(model="forest_c", params={"n_estimators": 10})
    assert tree_models(params) == "Train dataset is not yet split. Use 'split_data' first"

@patch("Autonoma.server.ModellingTools.tree_models")
def test_tree_models_success(mock_tool, mock_df):
    STATE['X_train'] = mock_df
    STATE['y_train'] = mock_df
    STATE['X_val'] = pl.DataFrame({"A": [5]})
    STATE['y_val'] = pl.DataFrame({"B": ["b"]})
    mock_tool.return_value = {"model_path": "path/to/tree"}
    params = TreeModelsSchema(model="forest_c", params={"n_estimators": 10})
    res = tree_models(params)
    assert res == "{'model_path': 'path/to/tree'}"
    assert STATE['model_path'] == "path/to/tree"
    mock_tool.assert_called_once_with(X_train=ANY, y_train=ANY, X_val=ANY, y_val=ANY, model="forest_c", params={"n_estimators": 10}, fit=True)

def test_eval_classification_no_data():
    params = EvalClassificationSchema(model_path="path/to/model")
    assert eval_classification(params) == "Test dataset is not yet split. Use 'split_data' first"

@patch("Autonoma.server.ModellingTools.eval_classification")
def test_eval_classification_with_test_data(mock_tool, mock_df):
    STATE['X_test'] = mock_df
    STATE['y_test'] = mock_df
    STATE['model_path'] = "path/to/model"
    mock_tool.return_value = {"accuracy": 0.9}
    params = EvalClassificationSchema(model_path="path/to/model")
    res = eval_classification(params)
    assert res == "{'accuracy': 0.9}"
    mock_tool.assert_called_once_with(X_test=ANY, y_test=ANY, model_path="path/to/model")

@patch("Autonoma.server.ModellingTools.eval_classification")
def test_eval_classification_with_val_data(mock_tool, mock_df):
    STATE['X_val'] = mock_df
    STATE['y_val'] = mock_df
    STATE['model_path'] = "path/to/model"
    mock_tool.return_value = {"accuracy": 0.8}
    params = EvalClassificationSchema(model_path="path/to/model")
    res = eval_classification(params)
    assert res == "{'accuracy': 0.8}"
    mock_tool.assert_called_once_with(X_test=ANY, y_test=ANY, model_path="path/to/model")

def test_eval_regression_no_data():
    params = EvalRegressionSchema(model_path="path/to/model")
    assert eval_regression(params) == "Test dataset is not yet split. Use 'split_data' first"

@patch("Autonoma.server.ModellingTools.eval_regression")
def test_eval_regression_with_test_data(mock_tool, mock_df):
    STATE['X_test'] = mock_df
    STATE['y_test'] = mock_df
    STATE['model_path'] = "path/to/model"
    mock_tool.return_value = {"mse": 0.1}
    params = EvalRegressionSchema(model_path="path/to/model")
    res = eval_regression(params)
    assert res == "{'mse': 0.1}"
    mock_tool.assert_called_once_with(X_test=ANY, y_test=ANY, model_path="path/to/model")

@patch("Autonoma.server.ModellingTools.eval_regression")
def test_eval_regression_with_val_data(mock_tool, mock_df):
    STATE['X_val'] = mock_df
    STATE['y_val'] = mock_df
    STATE['model_path'] = "path/to/model"
    mock_tool.return_value = {"mse": 0.2}
    params = EvalRegressionSchema(model_path="path/to/model")
    res = eval_regression(params)
    assert res == "{'mse': 0.2}"
    mock_tool.assert_called_once_with(X_test=ANY, y_test=ANY, model_path="path/to/model")
