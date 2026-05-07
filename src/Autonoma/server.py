# File:           server.py
# Description:    This file conatins the MCP server logic for autonoma
# Author:         Youssef Elebiary
# Date:           5/3/2026 - DD/MM/YYYY
# Version:        1.0



#############################
#     IMPORT LIBRARIES      #
#############################
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import polars as pl
from typing import (
    Dict,
    Optional
)

from mcp.server.fastmcp import FastMCP

from autonoma.tools.init import InitTools
from autonoma.tools.analysis import AnalysisTools
from autonoma.tools.preprocessing import PreprocessingTools
from autonoma.tools.modeling import ModellingTools

from autonoma.schema.tools_io import *
#############################



#############################
#       SERVER CONFIG       #
#############################
mcp = FastMCP("autonoma-MCP-Server")

STATE: Dict[str, Optional[pl.DataFrame]] = {
    "df": None,
    "X_train": None,
    "y_train": None,
    "X_val": None,
    "y_val": None,
    "X_test": None,
    "y_test": None,
    "model_path": ""
}
#############################



#############################
#     INIT STATE TOOLS      #
#############################
@mcp.tool()
def init_state(params: InitStateSchema) -> str:
    return InitTools.init_state(params.file_path, STATE)

#############################



#############################
#      ANALYSIS TOOLS       #
#############################
@mcp.tool()
def get_df_info() -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_df_info(STATE['df']))

@mcp.tool()
def get_numerical_summary(params: GetNumericalSummarySchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_numerical_summary(STATE['df'], **params.model_dump()))

@mcp.tool()
def get_categorical_distribution(params: GetCategoricalDistributionSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_categorical_distribution(STATE['df'], **params.model_dump()))

@mcp.tool()
def detect_outliers(params: DetectOutliersSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.detect_outliers(STATE['df'], **params.model_dump()))

@mcp.tool()
def get_correlation() -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_correlation(STATE['df']))

@mcp.tool()
def get_skew() -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_skew(STATE['df']))

@mcp.tool()
def get_target_correlations(params: GetTargetCorrelationSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_target_correlation(STATE['df'], **params.model_dump()))

@mcp.tool()
def get_column_cardinality(params: GetColumnCardinalitySchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    return str(AnalysisTools.get_column_cardinality(STATE['df'], **params.model_dump()))

@mcp.tool()
def check_low_variance(params: CheckLowVarianceSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    return str(AnalysisTools.check_low_variance(STATE['df'], **params.model_dump()))
#############################



#############################
#    PREPROCESSING TOOLS    #
#############################
@mcp.tool()
def drop_column(params: DropColumnSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"

    res, STATE['df'], STATE['X_train'], STATE['X_val'], STATE['X_test'] = PreprocessingTools.drop_column(
        STATE['df'], 
        column=params.column,
        X_train=STATE['X_train'],
        X_val=STATE['X_val'],
        X_test=STATE['X_test']
    )
    return str(res)

@mcp.tool()
def fill_nulls(params: FillNullsSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    res, STATE['df'], STATE['X_train'], STATE['X_val'], STATE['X_test'] = PreprocessingTools.fill_nulls(
        STATE['df'], 
        **params.model_dump(),
        X_train=STATE['X_train'],
        X_val=STATE['X_val'],
        X_test=STATE['X_test']
    )
    return str(res)

@mcp.tool()
def drop_outliers(params: DropOutliersSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    res, STATE['df'], STATE['X_train'], STATE['X_val'], STATE['X_test'] = PreprocessingTools.drop_outliers(
        STATE['df'], 
        **params.model_dump(),
        X_train=STATE['X_train'],
        X_val=STATE['X_val'],
        X_test=STATE['X_test']
    )
    return str(res)

@mcp.tool()
def transform_column(params: TransformColumnSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    res, STATE['df'], STATE['X_train'], STATE['X_val'], STATE['X_test'] = PreprocessingTools.transform_column(
        STATE['df'], 
        **params.model_dump(),
        X_train=STATE['X_train'],
        X_val=STATE['X_val'],
        X_test=STATE['X_test']
    )
    return str(res)

@mcp.tool()
def scale_column(params: ScaleColumnSchema) -> str:
    if STATE['X_train'] is not None:
        res, STATE['X_train'], STATE['X_test'], STATE['df'], STATE['X_val'] = PreprocessingTools.scale_column(
            STATE['X_train'], 
            STATE['X_test'], 
            **params.model_dump(),
            df=STATE['df'],
            X_val=STATE['X_val']
        )
        return str(res)
    else:
        return "Data has not been split. Use 'split_data' first"
    
@mcp.tool()
def encode_categorical(params: EncodeCategoricalSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    res, STATE['df'], STATE['X_train'], STATE['X_val'], STATE['X_test'] = PreprocessingTools.encode_categorical(
        STATE['df'], 
        **params.model_dump(),
        X_train=STATE['X_train'],
        y_train=STATE['y_train'],
        X_val=STATE['X_val'],
        X_test=STATE['X_test']
    )
    return str(res)

@mcp.tool()
def split_data(params: SplitDataSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    res, train, val, test = PreprocessingTools.split_data(STATE['df'], **params.model_dump())
    
    target = params.target
    STATE['X_train'] = train.drop(target)
    STATE['y_train'] = train.select(target)
    STATE['X_val'] = val.drop(target)
    STATE['y_val'] = val.select(target)
    if test is not None:
        STATE['X_test'] = test.drop(target)
        STATE['y_test'] = test.select(target)
    else:
        STATE['X_test'] = None
        STATE['y_test'] = None
        
    msg = f"{res['success']}. "
    msg += f"Train: {STATE['X_train'].shape}, Val: {STATE['X_val'].shape}"
    if STATE['X_test'] is not None:
        msg += f", Test: {STATE['X_test'].shape}"
    
    return msg

@mcp.tool()
def create_feature(params: CreateFeatureSchema) -> str:
    if STATE['df'] is None:
        return "No Data Loaded. Use 'init_state' first"
    
    res, STATE['df'], STATE['X_train'], STATE['X_val'], STATE['X_test'] = PreprocessingTools.create_feature(
        STATE['df'], 
        **params.model_dump(),
        X_train=STATE['X_train'],
        X_val=STATE['X_val'],
        X_test=STATE['X_test']
    )
    return str(res)
#############################



#############################
#      MODELING TOOLS       #
#############################
@mcp.tool()
def grid_search(params: GridSearchSchema) -> str:
    if STATE['X_train'] is not None and STATE['y_train'] is not None:
        return json.dumps(ModellingTools.grid_search(X_train=STATE['X_train'], y_train=STATE['y_train'], **params.model_dump()))
    else:
        return "Train dataset is not yet split. Use 'split_data' first"
    
@mcp.tool()
def random_search(params: RandomSearchSchema) -> str:
    if STATE['X_train'] is not None and STATE['y_train'] is not None:
        return json.dumps(ModellingTools.random_search(X_train=STATE['X_train'], y_train=STATE['y_train'], **params.model_dump()))
    else:
        return "Train dataset is not yet split. Use 'split_data' first"

@mcp.tool()
def linear_models(params: LinearModelsSchema) -> str:
    if STATE['X_train'] is not None and STATE['y_train'] is not None:
        res = ModellingTools.linear_models(X_train=STATE['X_train'], y_train=STATE['y_train'], **params.model_dump())
        STATE['model_path'] = res['model_path']
        return json.dumps(res)
    else:
        return "Train dataset is not yet split. Use 'split_data' first"

@mcp.tool()
def tree_models(params: TreeModelsSchema) -> str:
    if STATE['X_train'] is not None and STATE['y_train'] is not None:
        res = ModellingTools.tree_models(
            X_train=STATE['X_train'],
            y_train=STATE['y_train'],
            X_val=STATE['X_val'],
            y_val=STATE['y_val'],
            **params.model_dump()
        )
        STATE['model_path'] = res['model_path']
        return json.dumps(res)
    else:
        return "Train dataset is not yet split. Use 'split_data' first"

@mcp.tool()
def eval_classification(params: EvalClassificationSchema) -> str:
    if STATE['X_test'] is not None and STATE['y_test'] is not None:
        return json.dumps(ModellingTools.eval_classification(X_test=STATE['X_test'], y_test=STATE['y_test'], **params.model_dump()))
    elif STATE['X_val'] is not None and STATE['y_val'] is not None:
        return json.dumps(ModellingTools.eval_classification(X_test=STATE['X_val'], y_test=STATE['y_val'], **params.model_dump()))
    else:
        return "Test dataset is not yet split. Use 'split_data' first"

@mcp.tool()
def eval_regression(params: EvalRegressionSchema) -> str:
    if STATE['X_test'] is not None and STATE['y_test'] is not None:
        return json.dumps(ModellingTools.eval_regression(X_test=STATE['X_test'], y_test=STATE['y_test'], **params.model_dump()))
    elif STATE['X_val'] is not None and STATE['y_val'] is not None:
        return json.dumps(ModellingTools.eval_regression(X_test=STATE['X_val'], y_test=STATE['y_val'], **params.model_dump()))
    else:
        return "Test dataset is not yet split. Use 'split_data' first"
############################

if __name__ == "__main__":
    mcp.run(transport="stdio")