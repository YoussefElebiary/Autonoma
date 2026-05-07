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
    def drop_column(
        df: pl.DataFrame,
        column: str,
        X_train: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None,
        X_test: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df, X_train, X_val, X_test

        df = df.drop(column)
        if X_train is not None and column in X_train.columns:
            X_train = X_train.drop(column)
        if X_val is not None and column in X_val.columns:
            X_val = X_val.drop(column)
        if X_test is not None and column in X_test.columns:
            X_test = X_test.drop(column)
            
        return {"success": f"Column {column} dropped successfully."}, df, X_train, X_val, X_test

    @staticmethod
    def fill_nulls(
        df: pl.DataFrame,
        method: Literal['median', 'mode', 'below', 'knn', 'custom'],
        column: str,
        custom_val: Optional[str] = None,
        X_train: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None,
        X_test: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df, X_train, X_val, X_test
        
        fit_df = X_train if X_train is not None else df
        
        if method == 'median':
            val = fit_df.get_column(column).median()
        elif method == 'mode':
            val = fit_df.get_column(column).mode()[0]
        elif method == 'custom':
            if custom_val is None:
                return {"error": "Missing custom_val for method='custom'"}, df, X_train, X_val, X_test
            val = custom_val
        elif method == 'below':
            df = df.with_columns(pl.col(column).backward_fill())
            if X_train is not None: X_train = X_train.with_columns(pl.col(column).backward_fill())
            if X_val is not None: X_val = X_val.with_columns(pl.col(column).backward_fill())
            if X_test is not None: X_test = X_test.with_columns(pl.col(column).backward_fill())
            return {"success": "Imputation Successful"}, df, X_train, X_val, X_test
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            # KNN is more complex, usually fit on X_train
            if X_train is not None:
                X_train_np = X_train.to_numpy()
                col_idx = X_train.columns.index(column)
                imputer.fit(X_train_np)
                
                X_train = X_train.with_columns(pl.Series(column, imputer.transform(X_train_np)[:, col_idx]))
                if X_val is not None:
                    X_val = X_val.with_columns(pl.Series(column, imputer.transform(X_val.to_numpy())[:, col_idx]))
                if X_test is not None:
                    X_test = X_test.with_columns(pl.Series(column, imputer.transform(X_test.to_numpy())[:, col_idx]))
                # Update main df too
                df = df.with_columns(pl.Series(column, imputer.transform(df.to_numpy())[:, col_idx]))
            else:
                df_np = df.to_numpy()
                col_idx = df.columns.index(column)
                imputed = imputer.fit_transform(df_np)
                df = df.with_columns(pl.Series(column, imputed[:, col_idx]))
            return {"success": "Imputation Successful"}, df, X_train, X_val, X_test
        else:
            return {"error": f"Method {method} is invalid"}, df, X_train, X_val, X_test

        df = df.with_columns(pl.col(column).fill_null(val))
        if X_train is not None: X_train = X_train.with_columns(pl.col(column).fill_null(val))
        if X_val is not None: X_val = X_val.with_columns(pl.col(column).fill_null(val))
        if X_test is not None: X_test = X_test.with_columns(pl.col(column).fill_null(val))
        
        return {"success": f"Imputation Successful with value {val}"}, df, X_train, X_val, X_test
    
    @staticmethod
    def drop_outliers(
        df: pl.DataFrame,
        column: str,
        threshold: float = 3.0,
        X_train: Optional[pl.DataFrame] = None,
        y_train: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.DataFrame] = None,
        X_test: Optional[pl.DataFrame] = None,
        y_test: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df, X_train, y_train, X_val, y_val, X_test, y_test
        
        # Fit on X_train if available
        fit_df = X_train if X_train is not None else df
        mean = fit_df.get_column(column).mean()
        std = fit_df.get_column(column).std()
        
        if std is None or std < 1e-6:
            return {"error": "Data has zero variance"}, df, X_train, y_train, X_val, y_val, X_test, y_test
        
        def apply_filter(target_X, target_y=None):
            if target_X is None: return None, target_y
            mask = (target_X.get_column(column) - mean).abs() / std <= threshold
            target_X = target_X.filter(mask)
            if target_y is not None:
                target_y = target_y.filter(mask)
            return target_X, target_y

        mask_df = (df.get_column(column) - mean).abs() / std <= threshold
        df = df.filter(mask_df)

        X_train, y_train = apply_filter(X_train, y_train)
        X_val, y_val = apply_filter(X_val, y_val)
        X_test, y_test = apply_filter(X_test, y_test)

        return {
            "original_dim": fit_df.shape,
            "new_dim": df.shape,
            "threshold": threshold
        }, df, X_train, y_train, X_val, y_val, X_test, y_test
    
    @staticmethod
    def transform_column(
        df: pl.DataFrame,
        column: str,
        method: Literal['log', 'exp', 'box-cox', 'yeo-johnson'],
        X_train: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None,
        X_test: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df, X_train, X_val, X_test

        def apply_transform(target_df):
            if target_df is None: return None
            if method == 'log':
                return target_df.with_columns(pl.col(column).log1p())
            elif method == 'exp':
                return target_df.with_columns(pl.col(column).exp())
            elif method == 'box-cox':
                col_data = target_df.get_column(column).to_numpy()
                transformed = boxcox(col_data)[0]
                return target_df.with_columns(pl.Series(column, transformed))
            elif method == 'yeo-johnson':
                col_data = target_df.get_column(column).to_numpy()
                transformed = yeojohnson(col_data)[0]
                return target_df.with_columns(pl.Series(column, transformed))
            return target_df

        try:
            df = apply_transform(df)
            if X_train is not None: X_train = apply_transform(X_train)
            if X_val is not None: X_val = apply_transform(X_val)
            if X_test is not None: X_test = apply_transform(X_test)
            return {"success": "Transformation successful."}, df, X_train, X_val, X_test
        except Exception as e:
            return {"error": f"Transformation failed: {str(e)}"}, df, X_train, X_val, X_test
    
    @staticmethod
    def scale_column(
        X_train: pl.DataFrame,
        X_test: Optional[pl.DataFrame],
        columns: List[str],
        method: Literal['standard', 'minmax', 'maxabs', 'robust', 'quantile'] = 'standard',
        df: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        missing = [c for c in columns if c not in X_train.columns]
        if missing:
            return {"error": f"Columns {missing} not found in X_train."}, X_train, X_test, df, X_val

        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }

        if method not in scalers:
            return {"error": f"Method {method} is invalid."}, X_train, X_test, df, X_val

        scaler = scalers[method]

        train_scaled = scaler.fit_transform(X_train.select(columns).to_numpy())
        X_train = X_train.with_columns([pl.Series(col, train_scaled[:, i]) for i, col in enumerate(columns)])
        
        if X_test is not None:
            test_scaled = scaler.transform(X_test.select(columns).to_numpy())
            X_test = X_test.with_columns([pl.Series(col, test_scaled[:, i]) for i, col in enumerate(columns)])
            
        if X_val is not None:
            val_scaled = scaler.transform(X_val.select(columns).to_numpy())
            X_val = X_val.with_columns([pl.Series(col, val_scaled[:, i]) for i, col in enumerate(columns)])
            
        if df is not None:
            df_scaled = scaler.transform(df.select(columns).to_numpy())
            df = df.with_columns([pl.Series(col, df_scaled[:, i]) for i, col in enumerate(columns)])

        return {"success": "Scaling successful."}, X_train, X_test, df, X_val
    
    @staticmethod
    def encode_categorical(
        df: pl.DataFrame,
        column: str,
        method: Literal['onehot', 'label', 'ordinal', 'target', 'custom'],
        target_col: Optional[str],
        custom_map: Optional[Dict[str, int]],
        X_train: Optional[pl.DataFrame] = None,
        y_train: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None,
        X_test: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        if column not in df.columns:
            return {"error": f"Column {column} not found."}, df, X_train, X_val, X_test
        
        from sklearn.preprocessing import OneHotEncoder
        
        # Determine fit data
        fit_X = X_train if X_train is not None else df
        fit_y = y_train if y_train is not None else (df.select(target_col) if target_col and target_col in df.columns else None)

        try:
            if method == 'onehot':
                # Use sklearn for consistent columns across splits
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
                encoder.fit(fit_X.select(column).to_numpy())
                
                def apply_ohe(target_df):
                    if target_df is None: return None
                    encoded = encoder.transform(target_df.select(column).to_numpy())
                    # Get correct column names from categories
                    categories = encoder.categories_[0]
                    # We dropped the first category
                    cols = [f"{column}_{cat}" for cat in categories[1:]]
                    encoded_df = pl.DataFrame(encoded, schema=cols)
                    return pl.concat([target_df.drop(column), encoded_df], how="horizontal")

                df = apply_ohe(df)
                if X_train is not None: X_train = apply_ohe(X_train)
                if X_val is not None: X_val = apply_ohe(X_val)
                if X_test is not None: X_test = apply_ohe(X_test)

            elif method == 'label':
                # Polars categorical is enough if we just want numbers
                df = df.with_columns(pl.col(column).cast(pl.Categorical).to_physical().alias(column))
                if X_train is not None: X_train = X_train.with_columns(pl.col(column).cast(pl.Categorical).to_physical().alias(column))
                if X_val is not None: X_val = X_val.with_columns(pl.col(column).cast(pl.Categorical).to_physical().alias(column))
                if X_test is not None: X_test = X_test.with_columns(pl.col(column).cast(pl.Categorical).to_physical().alias(column))

            elif method == 'ordinal':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoder.fit(fit_X.select(column).to_numpy())
                
                def apply_ord(target_df):
                    if target_df is None: return None
                    encoded = encoder.transform(target_df.select(column).to_numpy())
                    return target_df.with_columns(pl.Series(column, encoded.flatten()))

                df = apply_ord(df)
                if X_train is not None: X_train = apply_ord(X_train)
                if X_val is not None: X_val = apply_ord(X_val)
                if X_test is not None: X_test = apply_ord(X_test)

            elif method == 'target':
                if target_col is None and y_train is None:
                    return {"error": "Missing target info for target encoding"}, df, X_train, X_val, X_test
                
                encoder = TargetEncoder()
                # If X_train exists, fit on it
                if X_train is not None and y_train is not None:
                    encoder.fit(X_train.select(column).to_numpy(), y_train.to_numpy().flatten())
                else:
                    encoder.fit(df.select(column).to_numpy(), df.get_column(target_col).to_numpy())

                def apply_tar(target_df):
                    if target_df is None: return None
                    encoded = encoder.transform(target_df.select(column).to_numpy())
                    return target_df.with_columns(pl.Series(column, encoded.flatten()))

                df = apply_tar(df)
                if X_train is not None: X_train = apply_tar(X_train)
                if X_val is not None: X_val = apply_tar(X_val)
                if X_test is not None: X_test = apply_tar(X_test)

            elif method == 'custom':
                if custom_map is None:
                    return {"error": "Missing custom_map for custom encoding"}, df, X_train, X_val, X_test
                df = df.with_columns(pl.col(column).replace(custom_map))
                if X_train is not None: X_train = X_train.with_columns(pl.col(column).replace(custom_map))
                if X_val is not None: X_val = X_val.with_columns(pl.col(column).replace(custom_map))
                if X_test is not None: X_test = X_test.with_columns(pl.col(column).replace(custom_map))
            else:
                return {"error": f"Method {method} is invalid."}, df, X_train, X_val, X_test

            return {"success": "Encoding successful."}, df, X_train, X_val, X_test
        except Exception as e:
            return {"error": f"Encoding failed: {str(e)}"}, df, X_train, X_val, X_test
    
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
        operations: List[Dict[str, Any]],
        X_train: Optional[pl.DataFrame] = None,
        X_val: Optional[pl.DataFrame] = None,
        X_test: Optional[pl.DataFrame] = None
    ) -> Tuple[Dict[str, Any], pl.DataFrame, Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """
        Applies a list of declarative feature engineering operations.
        Example operation: {"type": "arithmetic", "col1": "age", "col2": "tenure", "op": "mul", "new_name": "age_tenure_interaction"}
        """
        def apply_ops(target_df):
            if target_df is None: return None
            for op in operations:
                op_type = op.get('type')
                try:
                    if op_type == 'arithmetic':
                        col1, col2 = op['col1'], op['col2']
                        expr = getattr(pl.col(col1), op['op'])(pl.col(col2))
                        target_df = target_df.with_columns(expr.alias(op['new_name']))
                    elif op_type == 'date_extract':
                        col = op["col"]
                        target_df = target_df.with_columns([
                            pl.col(col).dt.month().alias(f"{col}_month"),
                            pl.col(col).dt.day().alias(f"{col}_day"),
                            pl.col(col).dt.weekday().alias(f"{col}_weekday")
                        ])
                except:
                    pass
            return target_df

        try:
            df = apply_ops(df)
            if X_train is not None: X_train = apply_ops(X_train)
            if X_val is not None: X_val = apply_ops(X_val)
            if X_test is not None: X_test = apply_ops(X_test)
            return {"success": "Feature creation successful"}, df, X_train, X_val, X_test
        except Exception as e:
            return {"error": f"Feature creation failed: {str(e)}"}, df, X_train, X_val, X_test

