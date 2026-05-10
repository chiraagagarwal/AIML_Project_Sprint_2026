CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "ridge_alphas": [0.1, 1.0, 10.0, 100.0],
    "lasso_alphas": [0.01, 0.1, 1.0, 10.0],
    "model_path": "house_price_pipeline/models/saved_model.pkl",
    "scaler_path": "house_price_pipeline/models/scaler.pkl",
}