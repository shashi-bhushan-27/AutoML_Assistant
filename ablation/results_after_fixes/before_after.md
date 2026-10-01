| Measure | Before (baseline) | After fixes |
|---|---|---|
| adult: stateful serving, worst agreement with offline (all batch sizes) | – | 1.000 |
| adult: stateful serving, worst crash rate | 1.000 | 0.000 |
| credit: stateful serving, worst agreement with offline (all batch sizes) | 0.914 | 1.000 |
| credit: stateful serving, worst crash rate | 0.000 | 0.000 |
| california: stateful serving, worst agreement with offline (all batch sizes) | – | 1.000 |
| california: stateful serving, worst crash rate | 1.000 | 0.000 |
| adult: /predict status codes (unperturbed + reordered requests) | 500×30 | 200×30 |
| credit: /predict status codes (unperturbed + reordered requests) | 200×15, 500×15 | 200×30 |
| california: /predict status codes (unperturbed + reordered requests) | 500×30 | 200×30 |
| adult: /predict single-row latency, XGBoost, median ms (TestClient round trip) | 68.7 | 12.6 |
| adult: … of which server-side (after only) | – | 4.6 |
| credit: /predict single-row latency, XGBoost, median ms (TestClient round trip) | 58.6 | 17.2 |
| credit: … of which server-side (after only) | – | 6.4 |
| california: /predict single-row latency, XGBoost, median ms (TestClient round trip) | 83.8 | 12.5 |
| california: … of which server-side (after only) | – | 3.5 |
| adult: pipeline pickle size (MB) | 5.38 | 0.017 |
| credit: pipeline pickle size (MB) | 72.80 | 0.028 |
| california: pipeline pickle size (MB) | 1.89 | 0.078 |
| adult: SHAP families with values (app routing, of 6) | 4.0 | 3.0 (6.0 incl. clean budget stop) |
| credit: SHAP families with values (app routing, of 6) | 3.2 | 4.0 (6.0 incl. clean budget stop) |
| california: SHAP families with values (app routing, of 6) | 5.0 | 4.8 (6.0 incl. clean budget stop) |
| adult: best single model (Accuracy / F1 Score) | Gradient Boosting 0.844 / 0.639 | Hist Gradient Boosting 0.873 / 0.709 |
| credit: best single model (Accuracy / F1 Score) | XGBoost 1.000 / 0.856 | XGBoost 1.000 / 0.855 |
| california: best single model (RMSE / R²) | XGBoost 0.460 / 0.839 | Hist Gradient Boosting 0.464 / 0.837 |
| adult: 'as shipped' advisor fallback rate | 100% | 0% |
| adult: full RAG, training-time reduction vs brute force | 69% | 82% |
| adult: meta-learning history in prompt (full / same-dataset prior) | 0% / 100% | 0% / 100% |
| credit: 'as shipped' advisor fallback rate | 100% | 0% |
| credit: full RAG, training-time reduction vs brute force | 32% | 76% |
| credit: meta-learning history in prompt (full / same-dataset prior) | 0% / 100% | 0% / 100% |
| california: 'as shipped' advisor fallback rate | 100% | 0% |
| california: full RAG, training-time reduction vs brute force | 32% | 36% |
| california: meta-learning history in prompt (full / same-dataset prior) | 0% / 0% | 0% / 100% |
| Name matcher on the 30 probe names | 27/60 names mapped or flagged; 'RandomForestClassifierRegressor' → substring | 58/60 names mapped or flagged; 'RandomForestClassifierRegressor' → unmatched |
| Meta-learning rule between the benchmarks | 0/6 cross-dataset pairs match | 0/6 cross-dataset pairs match; same-dataset prior matches for 3/3 |
