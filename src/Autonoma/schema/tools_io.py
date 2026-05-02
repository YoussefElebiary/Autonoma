from pydantic import BaseModel, Field
from typing import (
    Optional,
    List,
    Dict,
    Literal,
    Any
)



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
        default=None,
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
        default=10,
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
        default=3.0,
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
        default=0.95,
        description="Threshold (0.0 to 1.0). If the most frequent value occurs more than this fraction of the time, the feature is flagged."
    )
#############################



#############################
#       PREPROCESSING       #
#############################
class DropColumnSchema(BaseModel):
    """
    Drops a column from the dataset.
    Use this for features with too many missing values, zero variance, or data leakage.
    """
    column: str = Field(
        ...,
        description="The name of the column to remove from the DataFrame."
    )

class FillNullsSchema(BaseModel):
    """
    Imputes missing values (nulls) in a specific column.
    """
    method: Literal['median', 'mode', 'below', 'knn', 'custom'] = Field(
        ...,
        description="The imputation method. Use 'median' for skewed numerics, 'mode' for categoricals."
    )
    column: str = Field(
        ...,
        description="The name of the column to impute."
    )
    custom_val: Optional[str] = Field(
        default=None,
        description="The custom value to replace Nulls. Only required if method is 'custom'."
    )

class DropOutliersSchema(BaseModel):
    """
    Finds and removes extreme outliers in a numerical column using the Z-score method.
    """
    column: str = Field(
        ...,
        description="The name of the column to clean."
    )
    threshold: float = Field(
        default=3.0,
        description="The Z-score threshold. Values larger than this is dropped."
    )

class TransformColumnSchema(BaseModel):
    """
    Applies a mathematical transformation to a column to fix high skewness 
    and make its distribution more normal.
    """
    column: str = Field(
        ...,
        description="The name of the column to transform."
    )
    method: Literal['log', 'exp', 'box-cox', 'yeo-johnson'] = Field(
        ...,
        description="The transformation method. Use yeo-johnson if data contains zeros/negatives."
    )

class ScaleColumnSchema(BaseModel):
    """
    Applies feature scaling to numerical data. 
    MUST be applied AFTER splitting the data to prevent data leakage.
    """
    columns: List[str] = Field(
        ...,
        description="A list of the names of the numerical columns scale."
    )
    method: Literal['standard', 'minmax', 'maxabs', 'robust', 'quantile'] = Field(
        default='standard',
        description="Scaling method. Use 'robust' if outliers are still present."
    )

class EncodeCategoricalSchema(BaseModel):
    """
    Converts text/categorical columns into numerical formats required by machine learning models.
    Using Target method is ONLY allowed AFTER splitting the data to prevent data leakage.
    """
    column: str = Field(
        ...,
        description="The name of the categorical column to encode."
    )
    method: Literal['onehot', 'label', 'oridnal', 'target', 'custom'] = Field(
        ...,
        description="Encoding method. Use 'onehot' for low cardinality, 'target' for high cardinality."
    )
    target_col: Optional[str] = Field(
        default=None,
        description="The target column name. Only required if method is 'target'."
    )
    custom_map: Optional[Dict[str, int]] = Field(
        default=None,
        description="Dictionary mapping strings to ints. Only required if method is 'custom'."
    )

class SplitDataSchema(BaseModel):
    """
    Splits the dataset into Training, Validation, and Testing sets.
    MUST be called before any Scaling or Target Encoding operations.
    """
    target: str = Field(
        ...,
        description="The name of the target variable column."
    )
    val_size: float = Field(
        ...,
        description="Fraction of data for validation (e.g., 0.2)."
    )
    test_size: Optional[float] = Field(
        default=None,
        description="Optional fraction for the test dataset (e.g., 0.1)."
    )
    shuffle: bool = Field(
        default=True,
        description="Set to False when dealing with strictly sequential time-series data else set it to True."
    )
    seed: int = Field(
        default=42,
        description="The random state seed for reproducability"
    )

class CreateFeatureSchema(BaseModel):
    """
    Applies a list of declarative feature engineering operations.
    These 2 examples show the supported operations ONLY:
    Example #1: {"type": "arithmetic", "col1": "age", "col2": "tenure", "op": "mul", "new_name": "age_tenure_interaction"}
    Example #2: {"type": "date_extract", "col": "date_of_birth"}
    """
    operations: List[Dict[str, Any]] = Field(
        ...,
        description="A list of operation dictionaries specifying how to combine or extract features."
    )
#############################



#############################
#         MODELING          #
#############################

#############################