import os
import joblib
from datetime import datetime
import polars as pl
from typing import (
    Dict,
    Literal,
    Optional,
    Any
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import (
    XGBClassifier,
    XGBRegressor
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,

    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)

class ModellingTools:
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    @staticmethod
    def __save_model(model: Any, model_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{model_type}_{timestamp}.joblib"
        path = os.path.join(ModellingTools.MODEL_DIR, filename)
        joblib.dump(model, path)
        return path

    @staticmethod
    def __get_estimator(model_type: str) -> BaseEstimator:
        models = {
            'linear': LinearRegression,
            'logistic': LogisticRegression,
            'lasso': Lasso,
            'ridge': Ridge,
            'elastic': ElasticNet,
            'decision_c': DecisionTreeClassifier,
            'decision_r': DecisionTreeRegressor,
            'forest_c': RandomForestClassifier,
            'forest_r': RandomForestRegressor,
            'xgb_c': XGBClassifier,
            'xgb_r': XGBRegressor
        }
        if model_type not in models:
            raise ValueError(f"Invalid model_type: {model_type}")
        return models[model_type]()

    @staticmethod
    def grid_search(
        model_type: str,
        params: Dict[str, Any],
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        folds: int = 5
    ) -> Dict[str, Any]:
        grid = GridSearchCV(
            estimator=ModellingTools.__get_estimator(model_type),
            param_grid=params,
            n_jobs=-1,
            cv=folds
        )
        grid.fit(X_train.to_numpy(), y_train.to_numpy())
        return {
            "params": grid.best_params_,
            "score": grid.best_score_
        }
    
    @staticmethod
    def random_search(
        model_type: str,
        params: Dict[str, Any],
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        folds: int = 5,
        iters: int = 100
    ) -> Dict[str, Any]:
        randomized = RandomizedSearchCV(
            estimator=ModellingTools.__get_estimator(model_type),
            param_distributions=params,
            n_jobs=-1,
            cv=folds,
            n_iter=iters
        )
        randomized.fit(X_train.to_numpy(), y_train.to_numpy())
        return {
            "params": randomized.best_params_,
            "score": randomized.best_score_
        }
    
    @staticmethod
    def linear_models(
        model: Literal['linear', 'logistic', 'lasso', 'ridge', 'elastic'],
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        params: Dict[str, Any],
        fit: bool = True
    ) -> Dict[str, Any]:
        models = {
            'linear': lambda p: LinearRegression(**p),
            'logistic': lambda p: LogisticRegression(**p),
            'lasso': lambda p: Lasso(**p),
            'ridge': lambda p: Ridge(**p),
            'elastic': lambda p: ElasticNet(**p)
        }
        estimator = models[model](params)
        if fit:
            estimator.fit(X_train.to_numpy(), y_train.to_numpy())
        return {
            "model_path": ModellingTools.__save_model(estimator, model),
        }
    
    @staticmethod
    def tree_models(
        model: Literal['decision_c', 'decision_r', 'forest_c', 'forest_r', 'xgb_c', 'xgb_r'],
        X_train: pl.DataFrame,
        y_train: pl.DataFrame,
        X_val: Optional[pl.DataFrame],
        y_val: Optional[pl.DataFrame],
        params: Dict[str, Any],
        fit: bool = True
    ) -> Dict[str, Any]:
        if 'xgb' in model and X_val is not None and y_val is not None:
            params['early_stopping_rounds'] = 5

        models = {
            'decision_c': lambda p: DecisionTreeClassifier(**p),
            'decision_r': lambda p: DecisionTreeRegressor(**p),
            'forest_c': lambda p: RandomForestClassifier(**p),
            'forest_r': lambda p: RandomForestRegressor(**p),
            'xgb_c': lambda p: XGBClassifier(**p),
            'xgb_r': lambda p: XGBRegressor(**p)
        }
        estimator = models[model](params)
        if fit:
            if X_val is not None and y_val is not None and 'xgb' in model:
                estimator.fit(
                    X_train.to_numpy(),
                    y_train.to_numpy(),
                    eval_set = [(X_val.to_numpy(), y_val.to_numpy())],
                )
            else:
                estimator.fit(X_train.to_numpy(), y_train.to_numpy())
        return {
            "model_path": ModellingTools.__save_model(estimator, model),
        }
    
    @staticmethod
    def eval_classification(
        model_path: str,
        X_test: pl.DataFrame,
        y_test: pl.DataFrame,
    ) -> Dict[str, Any]:
        model = joblib.load(model_path)
        y_hat = model.predict(X_test.to_numpy())
        y_test_np = y_test.to_numpy()
        roc_auc = None
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test.to_numpy())
                if y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test_np, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test_np, y_prob, multi_class='ovr')
        except Exception:
            pass

        return {
            "accuracy": accuracy_score(y_test_np, y_hat),
            "recall": recall_score(y_test_np, y_hat, average='macro'),
            "precision": precision_score(y_test_np, y_hat, average='macro'),
            "f1": f1_score(y_test_np, y_hat, average='macro'),
            "roc_auc": roc_auc,
            "classification_report": classification_report(y_test_np, y_hat, output_dict=True)
        }
    
    @staticmethod
    def eval_regression(
        model_path: str,
        X_test: pl.DataFrame,
        y_test: pl.DataFrame,
    ) -> Dict[str, Any]:
        model = joblib.load(model_path)
        y_hat = model.predict(X_test.to_numpy())
        y_test_np = y_test.to_numpy()
        return {
            "mse": mean_squared_error(y_test_np, y_hat),
            "rmse": root_mean_squared_error(y_test_np, y_hat),
            "mae": mean_absolute_error(y_test_np, y_hat)
        }