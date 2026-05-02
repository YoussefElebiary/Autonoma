from pydantic import BaseModel, Field
from typing import Optional, List



#############################
#         ANALYSIS          #
#############################
class GetDFInfoSchema(BaseModel):
    """
    Get the basic structure of the DataFrame including dimensions, 
    column names, data types, and the number of Nulls. 
    Use this first to understand the dataset.
    """
    pass

class GetNumericalSummarySchema(BaseModel):
    """
    Calculates descriptive statistics (mean, std, min, max, percentiles) 
    for numerical columns to understand their distribution.
    """
    columns: Optional[List[str]] = Field(
        None,
        description="List of the numerical columns to summarize. If None, all columns are analyzed."
    )

class GetCategoricalSummarySchema(BaseModel):
    """
    Analyzes a categorical (string/object) column, returning the most 
    frequent values and their percentage of the total data.
    """
    column: str = Field(
        ...,
        description="The name of the categorical column to analyze" 
    )
    top_n: int = Field(
        10,
        description="The number of unique values to return"
    )

class DetectOutliersSchema(BaseModel):
    """
    Identifies outliers in a specific numerical column using the Z-Score method.
    Useful for finding anomalies before preprocessing.
    """
    column: str = Field(
        ...,
        description="The name of the column to find the outliers in using the Z-Score method",
    )
    threshold: float = Field(
        3.0,
        description="Z-score thershold for outlier detection using the Z-score method"
    )

class GetCorrelationSchema(BaseModel):
    """
    Generates a Pearson correlation matrix for all numerical columns. 
    Use this to detect multicollinearity between features.
    """
    pass

class GetSkewSchema(BaseModel):
    """
    Calculates the skewness of all numerical columns. 
    Use this to determine if transformations are needed.
    """
    pass

class GetTargetCorrelationSchema(BaseModel):
    """
    Calculates the correlation between all numerical features and a specific target column.
    Crucial for finding predictive features or detecting data leakage.
    """
    target: str = Field(
        ...,
        description="The name of the target column in the dataframe"
    )

class GetColumnCardinalitySchema(BaseModel):
    """
    Checks the number of unique values in a categorical column.
    Use this to decide between One-Hot Encoding and other encoding techniques.
    """
    column: str = Field(
        ...,
        description="The name of the column in the dataframe to check the cardinality of"
    )

class CheckLowVarianceSchema(BaseModel):
    """
    Detects quasi-constant features that provide almost no signal.
    """
    threshold: float = Field(
        0.95,
        description="Threshold (0.0 to 1.0). If the most frequent value occurs more than this fraction of the time, the feature is flagged."
    )
#############################



#############################
#       PREPROCESSING       #
#############################

#############################



#############################
#         MODELING          #
#############################

#############################