# File:           server.py
# Description:    This file conatins the MCP server logic for Autonoma
# Author:         Youssef Elebiary
# Date:           5/3/2026 - DD/MM/YYYY
# Version:        1.0



#############################
#     IMPORT LIBRARIES      #
#############################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import polars as pl
from typing import (
    Dict,
    Optional
)

from mcp.server.fastmcp import FastMCP


from Autonoma.tools.analysis import AnalysisTools
from Autonoma.tools.preprocessing import PreprocessingTools
from Autonoma.tools.modeling import ModellingTools

from Autonoma.schema.tools_io import *
#############################



#############################
#       SERVER CONFIG       #
#############################
mcp = FastMCP("Autonoma-MCP-Server")

STATE: Dict[str, Optional[pl.DataFrame]] = {
    "df": None,
    "X_train": None,
    "y_train": None,
    "X_val": None,
    "y_val": None,
    "X_test": None,
    "y_test": None
}

def init_state(file_path: str) -> None:
    global STATE
    df = pl.read_csv(file_path)
    STATE['df'] = df

init_state("D:\\Code\\Autonoma\\dirty_cafe_sales.csv")
#############################



#############################
#      ANALYSIS TOOLS       #
#############################
@mcp.tool()
def get_df_into() -> str:
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

if __name__ == "__main__":
    mcp.run(transport="stdio")