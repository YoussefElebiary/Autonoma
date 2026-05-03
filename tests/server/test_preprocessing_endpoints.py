import pytest
from unittest.mock import patch, ANY
import polars as pl
from Autonoma.server import (
    STATE,
    drop_column,
    fill_nulls,
    drop_outliers,
    transform_column,
    scale_column,
    encode_categorical,
    split_data,
    create_feature
)
from Autonoma.schema.tools_io import (
    DropColumnSchema,
    FillNullsSchema,
    DropOutliersSchema,
    TransformColumnSchema,
    ScaleColumnSchema,
    EncodeCategoricalSchema,
    SplitDataSchema,
    CreateFeatureSchema
)

def test_drop_column_no_data():
    params = DropColumnSchema(column="A")
    assert drop_column(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.drop_column")
def test_drop_column_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    new_df = pl.DataFrame({"B": ["a", "b", "a", "b", "a"]})
    mock_tool.return_value = ({"success": True}, new_df)
    params = DropColumnSchema(column="A")
    res = drop_column(params)
    assert res == "{'success': True}"
    assert STATE['df'].equals(new_df)
    mock_tool.assert_called_once_with(ANY, column="A")

def test_fill_nulls_no_data():
    params = FillNullsSchema(column="A", method="median")
    assert fill_nulls(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.fill_nulls")
def test_fill_nulls_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = ({"success": True}, mock_df)
    params = FillNullsSchema(column="A", method="median")
    res = fill_nulls(params)
    assert res == "{'success': True}"
    mock_tool.assert_called_once_with(ANY, column="A", method="median", custom_val=None)

def test_drop_outliers_no_data():
    params = DropOutliersSchema(column="A", threshold=3.0)
    assert drop_outliers(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.drop_outliers")
def test_drop_outliers_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = ({"success": True}, mock_df)
    params = DropOutliersSchema(column="A", threshold=3.0)
    res = drop_outliers(params)
    assert res == "{'success': True}"
    mock_tool.assert_called_once_with(ANY, column="A", threshold=3.0)

def test_transform_column_no_data():
    params = TransformColumnSchema(column="A", method="log")
    assert transform_column(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.transform_column")
def test_transform_column_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = ({"success": True}, mock_df)
    params = TransformColumnSchema(column="A", method="log")
    res = transform_column(params)
    assert res == "{'success': True}"
    mock_tool.assert_called_once_with(ANY, column="A", method="log")

def test_scale_column_no_data():
    params = ScaleColumnSchema(columns=["A"], method="standard")
    assert scale_column(params) == "No Data Loaded. Use 'init_state' first"

def test_scale_column_no_split(mock_df):
    STATE['df'] = mock_df
    params = ScaleColumnSchema(columns=["A"], method="standard")
    assert scale_column(params) == "Data has not been split. Use 'split_data' first"

@patch("Autonoma.server.PreprocessingTools.scale_column")
def test_scale_column_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    STATE['X_train'] = pl.DataFrame({"A": [1, 2]})
    STATE['X_test'] = pl.DataFrame({"A": [3]})
    new_train = pl.DataFrame({"A": [0, 1]})
    new_test = pl.DataFrame({"A": [2]})
    mock_tool.return_value = ({"success": True}, new_train, new_test)
    
    params = ScaleColumnSchema(columns=["A"], method="standard")
    res = scale_column(params)
    assert res == "{'success': True}"
    assert STATE['X_train'].equals(new_train)
    assert STATE['X_test'].equals(new_test)
    mock_tool.assert_called_once_with(ANY, ANY, columns=["A"], method="standard")

def test_encode_categorical_no_data():
    params = EncodeCategoricalSchema(column="B", method="onehot")
    assert encode_categorical(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.encode_categorical")
def test_encode_categorical_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = ({"success": True}, mock_df)
    params = EncodeCategoricalSchema(column="B", method="onehot")
    res = encode_categorical(params)
    assert res == "{'success': True}"
    mock_tool.assert_called_once_with(ANY, column="B", method="onehot", target_col=None, custom_map=None)

def test_split_data_no_data():
    params = SplitDataSchema(target="A", test_size=0.2, val_size=0.1)
    assert split_data(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.split_data")
def test_split_data_success_with_test(mock_tool, mock_df):
    STATE['df'] = mock_df
    train = pl.DataFrame({"A": [1, 2], "B": ["a", "b"]})
    val = pl.DataFrame({"A": [3], "B": ["a"]})
    test = pl.DataFrame({"A": [4], "B": ["b"]})
    mock_tool.return_value = ({"success": "Data split successfully"}, train, val, test)
    
    params = SplitDataSchema(target="A", test_size=0.2, val_size=0.1)
    res = split_data(params)
    
    assert res == "Data split successfully. Train: (2, 1), Val: (1, 1), Test: (1, 1)"
    assert STATE['X_train'].shape == (2, 1)
    assert STATE['y_train'].shape == (2, 1)
    assert STATE['X_test'].shape == (1, 1)

@patch("Autonoma.server.PreprocessingTools.split_data")
def test_split_data_success_without_test(mock_tool, mock_df):
    STATE['df'] = mock_df
    train = pl.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    val = pl.DataFrame({"A": [4], "B": ["d"]})
    mock_tool.return_value = ({"success": "Data split successfully"}, train, val, None)
    
    params = SplitDataSchema(target="A", test_size=0.0, val_size=0.25)
    res = split_data(params)
    
    assert res == "Data split successfully. Train: (3, 1), Val: (1, 1)"
    assert STATE['X_test'] is None
    assert STATE['y_test'] is None

def test_create_feature_no_data():
    params = CreateFeatureSchema(operations=[{"type": "arithmetic", "col1": "age", "col2": "tenure", "op": "mul", "new_name": "age_tenure_interaction"}])
    assert create_feature(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.PreprocessingTools.create_feature")
def test_create_feature_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = ({"success": True}, mock_df)
    params = CreateFeatureSchema(operations=[{"type": "arithmetic", "col1": "age", "col2": "tenure", "op": "mul", "new_name": "age_tenure_interaction"}])
    res = create_feature(params)
    assert res == "{'success': True}"
    mock_tool.assert_called_once_with(ANY, operations=[{"type": "arithmetic", "col1": "age", "col2": "tenure", "op": "mul", "new_name": "age_tenure_interaction"}])
