import numpy as np
import pandas as pd
import time
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from sklearn.base import clone
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

    def get_param_grid(self, model_name):
        """Returns the specific hyperparameter grid for a given model."""
        name = model_name.lower()
        
        # --- TREES & ENSEMBLES ---
        if "xgboost" in name:
            return {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        elif "random forest" in name or "extra trees" in name:
            return {
                'n_estimators': [100, 200, 400],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif "gradient boosting" in name:
            return {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.9, 1.0]
            }
        elif "adaboost" in name:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        elif "decision tree" in name:
            return {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # --- LINEAR MODELS ---
        elif "ridge" in name:
            return {'alpha': [0.1, 1.0, 10.0, 100.0]}
        elif "lasso" in name:
            return {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        elif "elasticnet" in name:
            return {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
        elif "logistic" in name:
            return {'C': [0.1, 1.0, 10.0], 'penalty': ['l2']}
        
        # --- OTHERS ---
        elif "svm" in name or "svr" in name or "svc" in name:
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        elif "knn" in name:
            return {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2] # 1=Manhattan, 2=Euclidean
            }
        
        return {} # No tuning available

    def tune_model(self, model_name):
        """
        Main routing function that picks the right tuning strategy.
        """
        start_time = time.time()
        print(f"Tuning {model_name}...")
        
        # 1. Special Handling for Prophet
        if model_name == "Prophet":
            return self.tune_prophet()
            
        # 2. Special Handling for ARIMA/SARIMAX
        if model_name in ["ARIMA", "SARIMAX"]:
            return self.tune_arima(model_name)

        # 3. Standard Scikit-Learn / XGBoost Tuning
        # Initialize base model
        model = self.get_base_model(model_name)
        if model is None:
            return {"Error": f"Could not find base model for {model_name}"}

        param_grid = self.get_param_grid(model_name)
        if not param_grid:
            return {"Error": f"No hyperparameter grid defined for {model_name}"}

        # Select CV Strategy
        if self.is_time_series:
            cv = TimeSeriesSplit(n_splits=3)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

        # metric selection
        scoring = 'neg_root_mean_squared_error' if self.task_type == "Regression" else 'accuracy'

        try:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=10, # Try 10 random combinations (Balance speed/quality)
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(self.X_train, self.y_train)
            best_model = search.best_estimator_
            
            # Save the best model back to the trainer
            tuned_name = f"{model_name} (Tuned)"
            self.trainer.trained_models[tuned_name] = best_model
            
            # Evaluate on Test Set
            preds = best_model.predict(self.trainer.X_test)
            return self.trainer.evaluate(self.trainer.y_test, preds, tuned_name, time.time() - start_time)
            
        except Exception as e:
            return {"Error": f"Tuning Failed: {str(e)}"}

    def get_base_model(self, name):
        """Helper to instantiate the correct class."""
        n = name.lower()
        is_reg = self.task_type == "Regression"
        
        if "xgboost" in n: return xgb.XGBRegressor(enable_categorical=False) if is_reg else xgb.XGBClassifier(enable_categorical=False)
        if "random forest" in n: return RandomForestRegressor() if is_reg else RandomForestClassifier()
        if "gradient boosting" in n: return GradientBoostingRegressor() if is_reg else GradientBoostingClassifier()
        if "adaboost" in n: return AdaBoostRegressor() if is_reg else AdaBoostClassifier()
        if "extra trees" in n: return ExtraTreesRegressor() if is_reg else ExtraTreesClassifier()
        if "decision tree" in n: return DecisionTreeRegressor() if is_reg else DecisionTreeClassifier()
        if "svm" in n: return SVR() if is_reg else SVC()
        if "knn" in n: return KNeighborsRegressor() if is_reg else KNeighborsClassifier()
        if "ridge" in n: return Ridge()
        if "lasso" in n: return Lasso()
        if "elasticnet" in n: return ElasticNet()
        if "logistic" in n: return LogisticRegression()
        return None

    def tune_prophet(self):
        """Custom tuning loop for Prophet (Grid Search)."""
        if not self.is_time_series: return {"Error": "Prophet requires Time Series data."}
        
        import itertools
        from sklearn.metrics import mean_squared_error
        
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
        }
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        
        best_params = None
        best_rmse = float('inf')
        best_model = None
        
        # Prepare Train Data
        train_df = self.trainer.df_train[[self.trainer.date_col, self.trainer.target_col]].rename(columns={self.trainer.date_col: 'ds', self.trainer.target_col: 'y'})
        # Validation Split (Last 20% of training data)
        cut = int(len(train_df) * 0.8)
        train_sub = train_df.iloc[:cut]
        val_sub = train_df.iloc[cut:]
        
        start = time.time()
        for params in all_params[:5]: # Limit to 5 combos for speed
            try:
                m = Prophet(**params)
                m.fit(train_sub)
                future = m.make_future_dataframe(periods=len(val_sub))
                forecast = m.predict(future)
                preds = forecast.iloc[-len(val_sub):]['yhat'].values
                rmse = np.sqrt(mean_squared_error(val_sub['y'], preds))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
            except: continue
            
        # Refit best model on FULL training data
        final_model = Prophet(**best_params)
        final_model.fit(train_df)
        
        tuned_name = "Prophet (Tuned)"
        self.trainer.trained_models[tuned_name] = final_model
        
        # Final Test
        future = final_model.make_future_dataframe(periods=len(self.trainer.df_test))
        forecast = final_model.predict(future)
        preds = forecast.iloc[-len(self.trainer.df_test):]['yhat'].values
        
        return self.trainer.evaluate(self.trainer.y_test, preds, tuned_name, time.time() - start)

    def tune_arima(self, model_name):
        """Simple Grid Search for ARIMA (p,d,q)."""
        import itertools
        from sklearn.metrics import mean_squared_error
        
        # Define p, d, q ranges
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        
        best_aic = float("inf")
        best_order = None
        
        start = time.time()
        for param in pdq[:6]: # Limit combos for speed
            try:
                if model_name == "SARIMAX":
                    mod = SARIMAX(self.trainer.y_train, order=param, seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
                else:
                    mod = ARIMA(self.trainer.y_train, order=param)
                
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
            except: continue
            
        # Refit best
        if model_name == "SARIMAX":
            final_model = SARIMAX(self.trainer.y_train, order=best_order, seasonal_order=(1,1,1,12)).fit(disp=False)
        else:
            final_model = ARIMA(self.trainer.y_train, order=best_order).fit()
            
        tuned_name = f"{model_name} (Tuned)"
        self.trainer.trained_models[tuned_name] = final_model # Note: statsmodels objects pickle differently
        
        preds = final_model.forecast(steps=len(self.trainer.y_test))
        return self.trainer.evaluate(self.trainer.y_test, preds, tuned_name, time.time() - start)