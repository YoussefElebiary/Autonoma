import polars as pl
import polars.selectors as cs
from typing import (
    List,
    Dict,
    Any,
    Optional
)

class AnalysisTools:
    @staticmethod
    def get_df_info(df: pl.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": df.columns,
            "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "null_counts": df.null_count().to_dicts()[0]
        }
    
    @staticmethod
    def get_numerical_summary(df: pl.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        target = df.select(columns) if columns is not None else df.select(cs.numeric())

        if target.width == 0:
            return {"error": "No Numerical Columns are Found or Selected"}

        summary = target.describe().to_dicts()
        return {"numerical_summary": summary}
    
    @staticmethod
    def get_categorical_distribution(df: pl.DataFrame, column: str, top_n: int = 10) -> Dict[str, Any]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}
            
        counts = (
            df.group_by(column)
            .agg(pl.len().alias("count"))
            .with_columns((pl.col("count") / df.height).alias("percentage"))
            .sort("count", descending=True)
            .head(top_n)
        )
        return {"column": column, "distribution": counts.to_dicts()}
    
    @staticmethod
    def detect_outliers(df: pl.DataFrame, column: str, threshold: float = 3.0) -> Dict[str, Any]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}

        data = df.select(pl.col(column))
        mean = data.mean().item()
        std = data.std().item()

        if abs(std) < 1e-6:
            return {
                "column": column,
                "outlier_count": 0,
                "note": "Zero Variance Column"
            }
        
        outliers = df.filter(((pl.col(column) - mean).abs() / std) > threshold)
        return {
            "column": column,
            "outlier_count": outliers.height,
            "threshold": threshold,
            "percentage": (outliers.height / df.height) * 100
        }
    
    @staticmethod
    def get_correlation(df: pl.DataFrame) -> Dict[str, Any]:
        num_cols = df.select(cs.numeric())
        if num_cols.width < 2:
            return {"error": "Insufficient Numerical Columns for Correlation"}
        
        corr = num_cols.corr()
        return {
            "matrix": corr.to_dicts(),
            "columns": num_cols.columns
        }
    
    @staticmethod
    def get_skew(df: pl.DataFrame) -> Dict[str, Any]:
        num_cols = df.select(cs.numeric())

        if num_cols.width <= 0:
            return {"error": "No Numerical Columns are Found or Selected"}
        
        skews = num_cols.select(pl.all().skew())

        return {
            "skews": skews.to_dicts(),
            "columns": num_cols.columns
        }
    
    @staticmethod
    def get_target_correlation(df: pl.DataFrame, target: str) -> Dict[str, Any]:
        if target not in df.columns:
            return {"error": f"Column {target} not found."}

        num_cols = df.select(cs.numeric())
        
        if target not in num_cols.columns:
            return {"error": f"Target {target} must be numerical to calculate Pearson correlation."}
            
        if num_cols.width < 2:
            return {"error": "Insufficient numerical columns for correlation."}

        correlations = {}
        for col in num_cols.columns:
            if col != target:
                corr_val = df.select(pl.corr(col, target)).item()
                correlations[col] = corr_val

        return {
            "target": target,
            "correlations": correlations
        }

    @staticmethod
    def get_column_cardinality(df: pl.DataFrame, column: str) -> Dict[str, Any]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}

        if df.schema[column] not in [pl.String, pl.Categorical]:
            return {"error": "Cannot check cardinality for numerical/boolean column"}
        
        return {
            "column": column,
            "n_unique": df.get_column(column).n_unique()
        }
    
    @staticmethod
    def check_low_variance(df: pl.DataFrame, threshold: float = 0.95) -> Dict[str, Any]:
        quasi_cols = []
        quasi_scores = []
        for col in df.columns:
            counts_df = df.get_column(col).value_counts()
            max_count = counts_df.get_column("count").max()
            most_freq = max_count / df.height

            if most_freq >= threshold:
                quasi_cols.append(col)
                quasi_scores.append(most_freq)

        return {
            "quasi_cols": quasi_cols,
            "quasi_scores": quasi_scores,
            "threshold": threshold
        }