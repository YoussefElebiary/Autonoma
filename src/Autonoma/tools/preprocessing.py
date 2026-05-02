import polars as pl
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Literal
)
from scipy.stats import boxcox, yeojohnson
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer
)
from sklearn.preprocessing import (
    OrdinalEncoder,
    TargetEncoder
)
from sklearn.model_selection import train_test_split

class PreprocessingTools:
    @staticmethod
    def drop_column(df: pl.DataFrame, column: str) -> Tuple[Dict[str, Any], pl.DataFrame]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df

        df = df.drop(column)
        return {"success": f"Column {column} dropped successfully."}, df

    @staticmethod
    def fill_nulls(
        df: pl.DataFrame,
        method: Literal['median', 'mode', 'below', 'knn', 'custom'],
        column: str,
        custom_val: Optional[str] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df
        
        if method == 'median':
            df = df.with_columns(pl.col(column).fill_null(pl.median(column)))
        elif method == 'mode':
            mode_val = df.get_column(column).mode()[0]
            df = df.with_columns(pl.col(column).fill_null(mode_val))
        elif method == 'below':
            df = df.with_columns(pl.col(column).backward_fill())
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            full_data = df.to_numpy()
            col_idx = df.columns.index(column)
            imputed_data = imputer.fit_transform(full_data)
            df = df.with_columns(pl.Series(column, imputed_data[:, col_idx]))
        elif method == 'custom':
            if custom_val is not None:
                df = df.with_columns(pl.col(column).fill_null(custom_val))
            else:
                return {"error": "Missing required custom_val parameter with method = custom"}, df
        else:
            return {"error": f"Method {method} is invalid"}, df
        
        return {"success": "Imputation Successful"}, df
    
    @staticmethod
    def drop_outliers(df: pl.DataFrame, column: str, threshold: float = 3.0) -> Tuple[Dict[str, Any], pl.DataFrame]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df
        
        mean = df.get_column(column).mean()
        std = df.get_column(column).std()
        
        if std is None or std < 1e-6:
            return {"error": "Data has zero variance"}, df
        
        filtered_data = df.filter((pl.col(column) - mean).abs() / std <= threshold)
        return {
            "original_dim": df.shape,
            "new_dim": filtered_data.shape,
            "threshold": threshold
        }, filtered_data
    
    @staticmethod
    def transform_column(
        df: pl.DataFrame,
        column: str,
        method: Literal['log', 'exp', 'box-cox', 'yeo-johnson']
    ) -> Tuple[Dict[str, Any], pl.DataFrame]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df

        if method == 'log':
            try:
                df = df.with_columns(pl.col(column).log1p())
            except Exception as e:
                return {"error": f"Log transformation failed with error {str(e)}"}, df
        elif method == 'exp':
            try:
                df = df.with_columns(pl.col(column).exp())
            except Exception as e:
                return {"error": f"Exp transformation failed with error {str(e)}"}, df
        elif method == 'box-cox':
            try:
                col_data = df.get_column(column).to_numpy()
                transformed = boxcox(col_data)[0]
                df = df.with_columns(pl.Series(column, transformed))
            except Exception as e:
                return {"error": f"Box-Cox transformation failed with error {str(e)}"}, df
        elif method == 'yeo-johnson':
            try:
                col_data = df.get_column(column).to_numpy()
                transformed = yeojohnson(col_data)[0]
                df = df.with_columns(pl.Series(column, transformed))
            except Exception as e:
                return {"error": f"Yeo-Johnson transformation failed with error {str(e)}"}, df
        else:
            return {"error": f"Method {method} is invalid."}, df

        return {"success": "Transformation successful."}, df
    
    @staticmethod
    def scale_column(
        X_train: pl.DataFrame,
        X_test: pl.DataFrame,
        columns: List[str],
        method: Literal['standard', 'minmax', 'maxabs', 'robust', 'quantile'] = 'standard'
    ) -> Tuple[Dict[str, Any], pl.DataFrame, pl.DataFrame]:
        missing = [c for c in columns if c not in X_train.columns or c not in X_test.columns]
        if missing:
            return {"error": f"Columns {missing} not found."}, X_train, X_test

        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }

        if method not in scalers:
            return {"error": f"Method {method} is invalid."}, X_train, X_test

        scaler = scalers[method]

        train_scaled = scaler.fit_transform(X_train.select(columns).to_numpy())
        test_scaled = scaler.transform(X_test.select(columns).to_numpy())

        X_train = X_train.with_columns([pl.Series(col, train_scaled[:, i]) for i, col in enumerate(columns)])
        X_test = X_test.with_columns([pl.Series(col, test_scaled[:, i]) for i, col in enumerate(columns)])

        return {"success": "Scaling successful."}, X_train, X_test
    
    @staticmethod
    def encode_categorical(
        df: pl.DataFrame,
        column: str,
        method: Literal['onehot', 'label', 'ordinal', 'target', 'custom'],
        target_col: Optional[str],
        custom_map: Optional[Dict[str, int]]
    ) -> Tuple[Dict[str, Any], pl.DataFrame]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df
        
        if method == 'onehot':
            df = df.to_dummies(columns=[column], drop_first=True)
        elif method == 'label':
            df = df.with_columns(pl.col(column).cast(pl.Categorical).to_physical().alias(column))
        elif method == 'ordinal':
            od = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded = od.fit_transform(df.select(column).to_numpy())
            df = df.with_columns(pl.Series(column, encoded.flatten()))
        elif method == 'target':
            if target_col is None:
                return {"error": "Missing required parameter target_col with method = target"}, df
            tar = TargetEncoder()
            encoded = tar.fit_transform(df.select(column).to_numpy(), df.get_column(target_col).to_numpy())
            df = df.with_columns(pl.Series(column, encoded.flatten()))
        elif method == 'custom':
            if custom_map is None:
                return {"error": "Missing required parameter custom_map with method = custom"}, df
            df = df.with_columns(pl.col(column).replace(custom_map))
        else:
            return {"error": f"Method {method} is invalid."}, df

        return {"success": "Scaling successful."}, df
    
    @staticmethod
    def split_data(
        df: pl.DataFrame,
        target: str,
        val_size: float,
        test_size: Optional[float],
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[Dict[str, Any], pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame]]:
        target_series = df.get_column(target)
        if test_size is not None:
            train, temp = train_test_split(
                df,
                test_size=val_size + test_size,
                shuffle=shuffle,
                stratify=df[target],
                random_state=seed
            )
            relative_test_size = test_size / (val_size + test_size)
            temp_target = temp.get_column(target)
            val, test = train_test_split(
                temp,
                test_size=relative_test_size,
                shuffle=shuffle,
                stratify=temp_target,
                random_state=seed
            )
            return {"success": "Train/Val/Test split successful"}, train, val, test
        else:
            train, val = train_test_split(
                df,
                test_size=val_size,
                shuffle=shuffle,
                stratify=target_series,
                random_state=seed
            )
            return {"success": "Train/Val split successful"}, train, val, None
        
    @staticmethod
    def create_feature(
        df: pl.DataFrame,
        operations: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], pl.DataFrame]:
        """
        Applies a list of declarative feature engineering operations.
        Example operation: {"type": "arithmetic", "col1": "age", "col2": "tenure", "op": "mul", "new_name": "age_tenure_interaction"}
        """
        for op in operations:
            op_type = op.get('type')

            try:
                if op_type == 'arithmetic':
                    col1, col2 = op['col1'], op['col2']
                    expr = getattr(pl.col(col1), op['op'])(pl.col(col2))
                    df = df.with_columns(expr.alias(op['new_name']))
                elif op_type == 'date_extract':
                    col = op["col"]
                    df = df.with_columns([
                        pl.col(col).dt.month().alias(f"{col}_month"),
                        pl.col(col).dt.day().alias(f"{col}_day"),
                        pl.col(col).dt.weekday().alias(f"{col}_weekday")
                    ])
                else:
                    return {"error": f"Operation type {op_type} is invalid."}, df
            except Exception as e:
                return {"error": f"Feature creation failed on {op_type}: {str(e)}"}, df
            
        return {"success": "Feature creation successful"}, df