import pandas as pd
import numpy as np
import time
import warnings
import importlib

warnings.filterwarnings('ignore')  # Silence warnings for cleaner UI

class ModelTrainer:
    def __init__(self, df, target_col, task_type='Regression', is_time_series=False, date_col=None):
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        self.is_time_series = is_time_series
        self.date_col = date_col
        self.results = []
        self.trained_models = {}
        self.predictions = {}
        self.prediction_probas = {}
        self.encoders = {} 
        self.scaler = None
        
        # Pre-processed data (set externally to skip internal preprocessing)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self._data_is_set = False
    
    def set_preprocessed_data(self, X_train, X_test, y_train, y_test):
        """Set pre-processed data to skip internal preprocessing."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self._data_is_set = True

    def get_supported_models(self):
        """Returns a list of all models this trainer supports."""
        common = ["XGBoost", "Random Forest", "Gradient Boosting", "AdaBoost", "Extra Trees", "Decision Tree", "SVM", "KNN"]
        if self.task_type == "Regression":
            common += ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]
        else:
            common += ["Logistic Regression"]

        if self.is_time_series:
            common += ["Prophet", "ARIMA", "SARIMAX"]
            # LSTM available only if tensorflow is installed
            try:
                import tensorflow  # noqa: F401
                common.append("LSTM")
            except ImportError:
                pass

        return sorted(common)

    def preprocess_data(self, X, target_col=None):
        """
        Uses AdvancedPreprocessor for intelligent type detection and encoding.
        """
        from app_backend.preprocessing import AdvancedPreprocessor
        
        self.preprocessor = AdvancedPreprocessor()
        X_processed = self.preprocessor.fit_transform(X, target_col=target_col)
        
        # Store encoders for potential inverse transform
        self.encoders = self.preprocessor.encoders
        
        # Scale data (Important for SVM, KNN, Linear models)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_processed = pd.DataFrame(self.scaler.fit_transform(X_processed), columns=X_processed.columns)
            
        return X_processed
    
    def get_preprocessing_report(self):
        """Returns the preprocessing report if available."""
        if hasattr(self, 'preprocessor'):
            return self.preprocessor.get_report()
        return None
    
    def get_preprocessing_summary(self):
        """Returns human-readable preprocessing summary."""
        if hasattr(self, 'preprocessor'):
            return self.preprocessor.get_summary_text()
        return "No preprocessing performed yet."

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

    def evaluate(self, y_true, y_pred, model_name, training_time, model_obj=None):
        """
        Evaluate model predictions and return a rich metrics dictionary.
        Handles both binary and multi-class classification with full metric suite.
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, f1_score, precision_score, recall_score,
            matthews_corrcoef, cohen_kappa_score,
            roc_auc_score, classification_report
        )
        from sklearn.preprocessing import label_binarize

        metrics = {"Model": model_name, "Time (s)": round(training_time, 4)}

        try:
            if self.task_type == "Regression":
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics["RMSE"] = round(rmse, 4)
                metrics["MAE"]  = round(mean_absolute_error(y_true, y_pred), 4)
                metrics["R²"]   = round(r2_score(y_true, y_pred), 4)
                # MAPE (avoid div-by-zero)
                non_zero = np.abs(y_true) > 1e-8
                if non_zero.any():
                    mape = np.mean(np.abs((np.array(y_true)[non_zero] - np.array(y_pred)[non_zero]) /
                                          np.abs(np.array(y_true)[non_zero]))) * 100
                    metrics["MAPE (%)"] = round(float(mape), 4)

            else:  # Classification
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                classes = np.unique(y_true_arr)
                is_binary = len(classes) == 2
                avg = 'binary' if is_binary else 'weighted'

                metrics["Accuracy"]  = round(accuracy_score(y_true_arr, y_pred_arr), 4)
                metrics["F1 Score"]  = round(f1_score(y_true_arr, y_pred_arr, average=avg, zero_division=0), 4)
                metrics["Precision"] = round(precision_score(y_true_arr, y_pred_arr, average=avg, zero_division=0), 4)
                metrics["Recall"]    = round(recall_score(y_true_arr, y_pred_arr, average=avg, zero_division=0), 4)

                # ── Enhanced Multi-Class Metrics ──────────────────────────
                try:
                    metrics["MCC"] = round(float(matthews_corrcoef(y_true_arr, y_pred_arr)), 4)
                except Exception:
                    pass

                try:
                    metrics["Cohen Kappa"] = round(float(cohen_kappa_score(y_true_arr, y_pred_arr)), 4)
                except Exception:
                    pass

                # AUC-ROC: binary uses proba[:, 1], multi-class uses OvR
                if model_obj is not None and hasattr(model_obj, 'predict_proba'):
                    try:
                        probas = model_obj.predict_proba(self.X_test)
                        if is_binary:
                            auc = roc_auc_score(y_true_arr, probas[:, 1])
                        else:
                            y_bin = label_binarize(y_true_arr, classes=classes)
                            auc = roc_auc_score(y_bin, probas, multi_class='ovr', average='weighted')
                        metrics["AUC-ROC"] = round(float(auc), 4)
                    except Exception:
                        pass

                # Per-class F1 report stored as JSON string for UI display
                try:
                    report_dict = classification_report(
                        y_true_arr, y_pred_arr, output_dict=True, zero_division=0
                    )
                    # Keep only per-class rows (exclude avg rows)
                    per_class = {k: v for k, v in report_dict.items()
                                 if k not in ('accuracy', 'macro avg', 'weighted avg')}
                    metrics["Per-Class Report"] = per_class
                except Exception:
                    pass

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

            # Store model + predictions for visualizations
            self.trained_models[name] = model
            self.predictions[name] = preds

            # Store predict_proba for ROC curves (classification only)
            if hasattr(model, 'predict_proba'):
                try:
                    self.prediction_probas[name] = model.predict_proba(self.X_test)
                except Exception:
                    pass

            return self.evaluate(self.y_test, preds, name, time.time() - start, model_obj=model)
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

    # --- LSTM (Deep Learning) ---
    def train_lstm(self):
        """
        Train a Keras LSTM network for univariate time-series forecasting.
        Requires tensorflow to be installed.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            return {"Model": "LSTM", "Error": "TensorFlow not installed. Run: pip install tensorflow"}

        if not self.is_time_series:
            return {"Model": "LSTM", "Error": "LSTM is only available for Time Series tasks."}

        start = time.time()
        try:
            LOOKBACK = 10  # Use last 10 timesteps to predict next value

            y_train_arr = np.array(self.y_train, dtype=np.float32)
            y_test_arr  = np.array(self.y_test,  dtype=np.float32)

            # Normalise
            y_min, y_max = y_train_arr.min(), y_train_arr.max()
            y_range = y_max - y_min + 1e-8
            y_train_norm = (y_train_arr - y_min) / y_range
            y_test_norm  = (y_test_arr  - y_min) / y_range

            # Build supervised sequences from y_train only
            def make_sequences(series, lookback):
                X_s, y_s = [], []
                for i in range(lookback, len(series)):
                    X_s.append(series[i - lookback: i])
                    y_s.append(series[i])
                return np.array(X_s)[..., np.newaxis], np.array(y_s)

            X_seq, y_seq = make_sequences(y_train_norm, LOOKBACK)
            if len(X_seq) < 5:
                return {"Model": "LSTM", "Error": "Not enough training samples for LSTM (need > LOOKBACK + 5)."}

            # Build model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X_seq, y_seq, epochs=50, batch_size=16,
                      callbacks=[es], verbose=0)

            # Walk-forward forecast on test set
            history = list(y_train_norm[-LOOKBACK:])
            preds_norm = []
            for _ in range(len(y_test_norm)):
                inp = np.array(history[-LOOKBACK:], dtype=np.float32)[np.newaxis, :, np.newaxis]
                out = float(model.predict(inp, verbose=0)[0, 0])
                preds_norm.append(out)
                history.append(out)

            preds = np.array(preds_norm) * y_range + y_min

            # Store a lightweight wrapper (keras model is not pickle-able by default)
            self.trained_models["LSTM"] = model
            self.predictions["LSTM"]    = preds

            return self.evaluate(y_test_arr, preds, "LSTM", time.time() - start)

        except Exception as e:
            return {"Model": "LSTM", "Error": str(e)}

    # --- MAIN CONTROLLER ---
    def run_selected_models(self, selected_model_names):
        # Skip preprocessing if data already set from AutoPreprocessor
        if not self._data_is_set:
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
            elif name == "LSTM":
                results.append(self.train_lstm())
                
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