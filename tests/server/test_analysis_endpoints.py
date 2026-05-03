import pytest
from unittest.mock import patch, MagicMock, ANY
from Autonoma.server import (
    STATE,
    get_df_into,
    get_numerical_summary,
    get_categorical_distribution,
    detect_outliers,
    get_correlation,
    get_skew,
    get_target_correlations,
    get_column_cardinality,
    check_low_variance
)
from Autonoma.schema.tools_io import (
    GetNumericalSummarySchema,
    GetCategoricalDistributionSchema,
    DetectOutliersSchema,
    GetTargetCorrelationSchema,
    GetColumnCardinalitySchema,
    CheckLowVarianceSchema
)

def test_get_df_into_no_data():
    assert get_df_into() == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_df_info")
def test_get_df_into_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"info": "success"}
    res = get_df_into()
    assert res == "{'info': 'success'}"
    mock_tool.assert_called_once_with(ANY)

def test_get_numerical_summary_no_data():
    params = GetNumericalSummarySchema(columns=["A"])
    assert get_numerical_summary(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_numerical_summary")
def test_get_numerical_summary_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"A": "summary"}
    params = GetNumericalSummarySchema(columns=["A"])
    res = get_numerical_summary(params)
    assert res == "{'A': 'summary'}"
    mock_tool.assert_called_once_with(ANY, columns=["A"])

def test_get_categorical_distribution_no_data():
    params = GetCategoricalDistributionSchema(column="B")
    assert get_categorical_distribution(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_categorical_distribution")
def test_get_categorical_distribution_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"B": "distribution"}
    params = GetCategoricalDistributionSchema(column="B")
    res = get_categorical_distribution(params)
    assert res == "{'B': 'distribution'}"
    mock_tool.assert_called_once_with(ANY, column="B", top_n=10)

def test_detect_outliers_no_data():
    params = DetectOutliersSchema(column="A", method="zscore", threshold=3.0)
    assert detect_outliers(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.detect_outliers")
def test_detect_outliers_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"A": "outliers"}
    params = DetectOutliersSchema(column="A", threshold=3.0)
    res = detect_outliers(params)
    assert res == "{'A': 'outliers'}"
    mock_tool.assert_called_once_with(ANY, column="A", threshold=3.0)

def test_get_correlation_no_data():
    assert get_correlation() == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_correlation")
def test_get_correlation_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"corr": 1.0}
    res = get_correlation()
    assert res == "{'corr': 1.0}"
    mock_tool.assert_called_once_with(ANY)

def test_get_skew_no_data():
    assert get_skew() == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_skew")
def test_get_skew_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"skew": 0.5}
    res = get_skew()
    assert res == "{'skew': 0.5}"
    mock_tool.assert_called_once_with(ANY)

def test_get_target_correlations_no_data():
    params = GetTargetCorrelationSchema(target="A")
    assert get_target_correlations(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_target_correlation")
def test_get_target_correlations_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"A": 1.0}
    params = GetTargetCorrelationSchema(target="A")
    res = get_target_correlations(params)
    assert res == "{'A': 1.0}"
    mock_tool.assert_called_once_with(ANY, target="A")

def test_get_column_cardinality_no_data():
    params = GetColumnCardinalitySchema(column="B")
    assert get_column_cardinality(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.get_column_cardinality")
def test_get_column_cardinality_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"B": 2}
    params = GetColumnCardinalitySchema(column="B")
    res = get_column_cardinality(params)
    assert res == "{'B': 2}"
    mock_tool.assert_called_once_with(ANY, column="B")

def test_check_low_variance_no_data():
    params = CheckLowVarianceSchema(threshold=0.01)
    assert check_low_variance(params) == "No Data Loaded. Use 'init_state' first"

@patch("Autonoma.server.AnalysisTools.check_low_variance")
def test_check_low_variance_success(mock_tool, mock_df):
    STATE['df'] = mock_df
    mock_tool.return_value = {"low_var": []}
    params = CheckLowVarianceSchema(threshold=0.01)
    res = check_low_variance(params)
    assert res == "{'low_var': []}"
    mock_tool.assert_called_once_with(ANY, threshold=0.01)
