import numpy as np
import pandas as pd
import time
import optuna
import logging
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- IMPORTS FROM YOUR TRAINER (To reuse classes) ---
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTuner:
    def __init__(self, trainer):
        """
        trainer: The existing ModelTrainer instance (already has X_train, y_train processed).
        """
        self.trainer = trainer
        self.X_train = trainer.X_train
        self.y_train = trainer.y_train
        self.task_type = trainer.task_type
        self.is_time_series = trainer.is_time_series

    def tune_model(self, model_name, time_budget=60, cv_folds=3, custom_params=None):
        """
        Main routing function that uses Optuna for Bayesian Optimization.
        Args:
            model_name: Name of the model to tune.
            time_budget: Max time in seconds for optimization.
            cv_folds: Number of cross-validation folds.
            custom_params: Ignored in Optuna for now, we use search spaces.
        """
        start_time = time.time()
        print(f"🚀 Tuning {model_name} with Optuna (Budget: {time_budget}s, CV: {cv_folds})...")
        
        # 1. Special Handling for Prophet
        if model_name == "Prophet":
            return self.tune_prophet(time_budget, cv_folds)
            
        # 2. Special Handling for ARIMA/SARIMAX
        if model_name in ["ARIMA", "SARIMAX"]:
            return self.tune_arima(model_name, time_budget)

        # 3. Standard Scikit-Learn / XGBoost Tuning
        
        # Objective function for Optuna
        def objective(trial):
            # Define search space dynamically
            params = self.get_optuna_params(trial, model_name)
            
            # Instantiate model
            model = self.get_base_model(model_name, params)
            if model is None:
                raise ValueError(f"Unknown model: {model_name}")
            
            # CV Strategy
            if self.is_time_series:
                cv = TimeSeriesSplit(n_splits=cv_folds)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
            scoring = 'neg_root_mean_squared_error' if self.task_type == "Regression" else 'accuracy'
            
            try:
                # Run CV
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=-1)
                return scores.mean()
            except Exception as e:
                # Prune bad trials
                raise optuna.exceptions.TrialPruned(str(e))

        # Optimization Direction
        direction = 'maximize' # Accuracy or NegRMSE (higher is better)
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, timeout=time_budget) # Auto-stop after time_budget seconds
        
        if len(study.trials) == 0:
            return {"Error": "Time budget too small, no trials completed."}

        # Retrieve best results
        best_params = study.best_params
        print(f"✅ Best Params: {best_params}")
        
        # Train Final Model with Best Params
        final_model = self.get_base_model(model_name, best_params)
        final_model.fit(self.X_train, self.y_train)
        
        # Save the best model back to the trainer
        tuned_name = f"{model_name} (Tuned)"
        self.trainer.trained_models[tuned_name] = final_model
        
        # Evaluate on Test Set
        preds = final_model.predict(self.trainer.X_test)
        
        eval_res = self.trainer.evaluate(self.trainer.y_test, preds, tuned_name, time.time() - start_time)
        eval_res["trials"] = len(study.trials)
        eval_res["Best Params"] = best_params
        return eval_res

    def get_optuna_params(self, trial, model_name):
        """Define Bayesian search space for each model."""
        n = model_name.lower()
        
        if "xgboost" in n:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'enable_categorical': False
            }
        elif "random forest" in n or "extra trees" in n:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif "gradient boosting" in n:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
        elif "svm" in n:
            return {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
        elif "knn" in n:
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            }
        elif "ridge" in n:
            return {'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)}
        elif "lasso" in n:
            return {'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True)}
        elif "elasticnet" in n:
            return {
                'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
            }
        elif "logistic" in n:
            return {'C': trial.suggest_float('C', 0.1, 100, log=True)}
        
        return {}

    def get_base_model(self, name, params):
        """Instantiate model with specific params."""
        n = name.lower()
        is_reg = self.task_type == "Regression"
        
        if "xgboost" in n: return xgb.XGBRegressor(**params) if is_reg else xgb.XGBClassifier(**params)
        if "random forest" in n: return RandomForestRegressor(**params) if is_reg else RandomForestClassifier(**params)
        if "gradient boosting" in n: return GradientBoostingRegressor(**params) if is_reg else GradientBoostingClassifier(**params)
        if "extra trees" in n: return ExtraTreesRegressor(**params) if is_reg else ExtraTreesClassifier(**params)
        if "svm" in n: return SVR(**params) if is_reg else SVC(**params)
        if "knn" in n: return KNeighborsRegressor(**params) if is_reg else KNeighborsClassifier(**params)
        if "ridge" in n: return Ridge(**params)
        if "lasso" in n: return Lasso(**params)
        if "elasticnet" in n: return ElasticNet(**params)
        if "logistic" in n: return LogisticRegression(**params)
        return None  

    def tune_prophet(self, time_budget, cv_folds):
        """Optuna tuning for Prophet."""
        if not self.is_time_series: return {"Error": "Prophet requires Time Series data."}
        
        from sklearn.metrics import mean_squared_error
        
        # Prepare Train Data
        train_df = self.trainer.df_train[[self.trainer.date_col, self.trainer.target_col]].rename(columns={self.trainer.date_col: 'ds', self.trainer.target_col: 'y'})
        cut = int(len(train_df) * 0.8)
        train_sub = train_df.iloc[:cut]
        val_sub = train_df.iloc[cut:]
        
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
            }
            try:
                m = Prophet(**params)
                m.fit(train_sub)
                future = m.make_future_dataframe(periods=len(val_sub))
                forecast = m.predict(future)
                preds = forecast.iloc[-len(val_sub):]['yhat'].values
                rmse = np.sqrt(mean_squared_error(val_sub['y'], preds))
                return rmse
            except:
                return float('inf')

        # Minimize RMSE
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, timeout=time_budget)
        
        best_params = study.best_params
        
        # Refit best
        final_model = Prophet(**best_params)
        final_model.fit(train_df)
        
        tuned_name = "Prophet (Tuned)"
        self.trainer.trained_models[tuned_name] = final_model
        
        future = final_model.make_future_dataframe(periods=len(self.trainer.df_test))
        forecast = final_model.predict(future)
        preds = forecast.iloc[-len(self.trainer.df_test):]['yhat'].values
        
        res = self.trainer.evaluate(self.trainer.y_test, preds, tuned_name, 0)
        res["Best Params"] = best_params
        return res

    def tune_arima(self, model_name, time_budget):
        # Simplified ARIMA (Optuna for ARIMA order is tricky, keeping simple search but with timeout check)
        # For now, return basic Grid implementation but respecting timeout roughly
        return {"Error": "ARIMA Optuna tuning not fully implemented yet due to complexity. Use Random Forest/XGB for now."}