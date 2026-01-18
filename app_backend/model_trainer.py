import pandas as pd
import numpy as np
import time
import warnings
import importlib

warnings.filterwarnings('ignore') # Silence warnings for cleaner UI

class ModelTrainer:
    def __init__(self, df, target_col, task_type='Regression', is_time_series=False, date_col=None):
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        self.is_time_series = is_time_series
        self.date_col = date_col
        self.results = []
        self.trained_models = {}
        self.encoders = {} 
        self.scaler = None

    def get_supported_models(self):
        """Returns a list of all models this trainer supports."""
        common = ["XGBoost", "Random Forest", "Gradient Boosting", "AdaBoost", "Extra Trees", "Decision Tree", "SVM", "KNN"]
        if self.task_type == "Regression":
            common += ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]
        else:
            common += ["Logistic Regression"]
            
        if self.is_time_series:
            common += ["Prophet", "ARIMA", "SARIMAX"]
            
        return sorted(common)

    def preprocess_data(self, X):
        # Lazy Import Sklearn Preprocessing
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.impute import SimpleImputer
        
        X_processed = X.copy()
        
        # 1. Encode Categoricals
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.encoders[col] = le
            
        # 2. Impute Missing Values (Safety Net)
        if X_processed.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_processed = pd.DataFrame(imputer.fit_transform(X_processed), columns=X_processed.columns)

        # 3. Scale Data (Important for SVM, KNN, Linear)
        self.scaler = StandardScaler()
        X_processed = pd.DataFrame(self.scaler.fit_transform(X_processed), columns=X_processed.columns)
            
        return X_processed

    def split_data(self):
        # Lazy Import Sklearn Selection
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        if self.date_col and self.date_col in X.columns:
            X = X.drop(columns=[self.date_col])

        X = self.preprocess_data(X)

        if self.task_type == "Classification" and y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))

        if self.is_time_series:
            train_size = int(len(self.df) * 0.8)
            self.X_train, self.X_test = X.iloc[:train_size], X.iloc[train_size:]
            self.y_train, self.y_test = y[:train_size], y[train_size:]
            self.df_train = self.df.iloc[:train_size]
            self.df_test = self.df.iloc[train_size:]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate(self, y_true, y_pred, model_name, training_time):
        # Lazy Import Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
        
        metrics = {"Model": model_name, "Time (s)": round(training_time, 4)}
        try:
            if self.task_type == "Regression":
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics["RMSE"] = round(rmse, 4)
                metrics["MAE"] = round(mean_absolute_error(y_true, y_pred), 4)
            else:
                metrics["Accuracy"] = round(accuracy_score(y_true, y_pred), 4)
                metrics["F1 Score"] = round(f1_score(y_true, y_pred, average='weighted'), 4)
        except Exception as e:
            metrics["Error"] = f"Eval Failed: {str(e)}"
        return metrics

    # --- GENERIC TRAINER FUNCTION ---
    def train_sklearn_model(self, name, model_class, **kwargs):
        start = time.time()
        try:
            model = model_class(**kwargs)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            return self.evaluate(self.y_test, preds, name, time.time() - start)
        except Exception as e:
            return {"Model": name, "Error": str(e)}

    # --- SPECIFIC TIME SERIES MODELS ---
    def train_prophet(self):
        # Lazy Import Prophet
        from prophet import Prophet
        
        start = time.time()
        if not self.is_time_series: return None
        try:
            train_df = self.df_train[[self.date_col, self.target_col]].rename(columns={self.date_col: 'ds', self.target_col: 'y'})
            model = Prophet()
            model.fit(train_df)
            future = model.make_future_dataframe(periods=len(self.df_test))
            forecast = model.predict(future)
            preds = forecast.iloc[-len(self.df_test):]['yhat'].values
            return self.evaluate(self.y_test, preds, "Prophet", time.time() - start)
        except Exception as e:
            return {"Model": "Prophet", "Error": str(e)}

    def train_arima(self, order=(1,1,1)):
        # Lazy Import ARIMA
        from statsmodels.tsa.arima.model import ARIMA
        
        start = time.time()
        if not self.is_time_series: return None
        try:
            # ARIMA needs strict 1D array
            model = ARIMA(self.y_train, order=order)
            model_fit = model.fit()
            preds = model_fit.forecast(steps=len(self.y_test))
            return self.evaluate(self.y_test, preds, "ARIMA", time.time() - start)
        except Exception as e:
            return {"Model": "ARIMA", "Error": str(e)}

    def train_sarimax(self):
        # Lazy Import SARIMAX
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        start = time.time()
        if not self.is_time_series: return None
        try:
            # Simple default config for MVP
            model = SARIMAX(self.y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit(disp=False)
            preds = model_fit.forecast(steps=len(self.y_test))
            return self.evaluate(self.y_test, preds, "SARIMAX", time.time() - start)
        except Exception as e:
            return {"Model": "SARIMAX", "Error": str(e)}

    # --- MAIN CONTROLLER ---
    def run_selected_models(self, selected_model_names):
        self.split_data()
        results = []
        
        # --- LAZY IMPORT MODELS ---
        import xgboost as xgb
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.svm import SVR, SVC
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        
        # Dictionary Mapping Names to Classes
        regressors = {
            "Linear Regression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "Random Forest": RandomForestRegressor,
            "XGBoost": xgb.XGBRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
            "AdaBoost": AdaBoostRegressor,
            "Extra Trees": ExtraTreesRegressor,
            "Decision Tree": DecisionTreeRegressor,
            "SVM": SVR,
            "KNN": KNeighborsRegressor
        }
        
        classifiers = {
            "Logistic Regression": LogisticRegression,
            "Random Forest": RandomForestClassifier,
            "XGBoost": xgb.XGBClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "AdaBoost": AdaBoostClassifier,
            "Extra Trees": ExtraTreesClassifier,
            "Decision Tree": DecisionTreeClassifier,
            "SVM": SVC,
            "KNN": KNeighborsClassifier
        }

        # Select the right dictionary
        models_map = regressors if self.task_type == "Regression" else classifiers

        for name in selected_model_names:
            print(f"Training {name}...")
            
            # 1. Handle Special Time Series Models
            if name == "Prophet":
                results.append(self.train_prophet())
            elif name == "ARIMA":
                results.append(self.train_arima())
            elif name == "SARIMAX":
                results.append(self.train_sarimax())
                
            # 2. Handle Standard Sklearn Models
            elif name in models_map:
                # XGBoost special handling for categorization
                if "XGBoost" in name:
                     results.append(self.train_sklearn_model(name, models_map[name], enable_categorical=False))
                else:
                     results.append(self.train_sklearn_model(name, models_map[name]))
            
            else:
                print(f"Warning: {name} not found in model map.")

        # Filter out Nones or failures
        clean_results = [r for r in results if r is not None]
        return pd.DataFrame(clean_results)

    def tune_specific_model(self, model_name):
        # Placeholder for simplified tuning
        return {"Error": "Tuning temporarily simplified for this update."}