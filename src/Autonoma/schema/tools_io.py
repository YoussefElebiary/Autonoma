from pydantic import BaseModel, Field
from typing import Optional, List



#############################
#         ANALYSIS          #
#############################
class GetDFInfoSchema(BaseModel):
    """
    Get the basic information about the DataFrame including the dimensions,
    column names, data types, and the number of Nulls.
    """
    pass

class GetNumericalSummarySchema(BaseModel):
    columns: Optional[List[str]] = Field(
        None,
        description="List of the numerical columns to summarize. If None, all columns are analyzed."
    )

class GetCategoricalSummarySchema(BaseModel):
    column: str = Field(
        ...,
        description="The name of the categorical column to analyze" 
    )
    top_n: int = Field(
        10,
        description="The number of unique values to return"
    )

class DetectOutliersSchema(BaseModel):
    column: str = Field(
        ...,
        description="The name of the column to find the outliers in using the Z-Score method",
    )
    threshold: float = Field(
        3.0,
        description="Z-score thershold for outlier detection using the Z-score method"
    )

class GetCorrelationSchema(BaseModel):
    pass

class GetSkewSchema(BaseModel):
    pass
#############################



#############################
#       PREPROCESSING       #
#############################

#############################



#############################
#         MODELING          #
#############################

#############################