import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from config import CONFIG

#Data Ingestion

raw = fetch_california_housing(as_frame=True)
df = raw.frame
df = df[df["MedHouseVal"] < 5.0].copy() 
print(f"  Shape after removing capped prices: {df.shape}")
print(df.describe().round(2))

#Feature engineering

df["rooms_per_household"]   = df["AveRooms"] / df["HouseAge"].clip(lower=1)
df["bedrooms_per_room"]     = df["AveBedrms"] / df["AveRooms"].clip(lower=1)
df["population_per_household"] = df["Population"] / df["AveOccup"].clip(lower=1)
df["log_median_income"]     = np.log1p(df["MedInc"])
# used log price for better linear fit, invert at prediction time
df["log_price"] = np.log1p(df["MedHouseVal"])

feature_cols = [
    "MedInc", "log_median_income", "HouseAge", "AveRooms",
    "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude",
    "rooms_per_household", "bedrooms_per_room", "population_per_household"
]
X = df[feature_cols]
y = df["log_price"]
print(f"  Features: {X.shape[1]} columns")

#Prepeocessing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"]
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   
X_test_s  = scaler.transform(X_test)        
print(f"  Train: {X_train_s.shape[0]} rows | Test: {X_test_s.shape[0]} rows")

#Model Training with GridSearchCV

candidates = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"alpha": CONFIG["ridge_alphas"]}),
    "Lasso": (Lasso(max_iter=5000), {"alpha": CONFIG["lasso_alphas"]}),
}

results = {}
for name, (model, params) in candidates.items():
    if params:
        gs = GridSearchCV(model, params, cv=CONFIG["cv_folds"],
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        best = gs.best_estimator_
        print(f"  {name}: best params = {gs.best_params_}")
    else:
        best = model.fit(X_train_s, y_train)
    results[name] = best

#Evaluation

best_model = list(results.values())[0]
best_rmse = float("inf")
best_name = list(results.keys())[0]

for name, model in results.items():
    preds_log = model.predict(X_test_s)
    preds     = np.expm1(preds_log)        # invert log transform
    actuals   = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds)
    cv_scores = cross_val_score(model, X_train_s, y_train,
                                cv=CONFIG["cv_folds"],
                                scoring="neg_root_mean_squared_error")
    cv_rmse = -cv_scores.mean()

    print(f"\n  {name}")
    print(f"    Test RMSE:  ${rmse * 100_000:,.0f}")
    print(f"    Test R²:    {r2:.4f}")
    print(f"    CV RMSE:    ${cv_rmse * 100_000:,.0f}  (±{cv_scores.std() * 100_000:,.0f})")

    if rmse < best_rmse:
        best_rmse, best_model, best_name = rmse, model, name

print(f"\n  Winner: {best_name} (RMSE ${best_rmse * 100_000:,.0f})")

#Residual Plot

preds_log = best_model.predict(X_test_s)
residuals = np.expm1(y_test) - np.expm1(preds_log)
plt.figure(figsize=(8, 4))
plt.scatter(np.expm1(preds_log), residuals, alpha=0.3, s=8)
plt.axhline(0, color="red", linewidth=1)
plt.xlabel("Predicted price ($100k)")
plt.ylabel("Residual")
plt.title(f"Residuals — {best_name}")
plt.tight_layout()
plt.savefig("house_price_pipeline/predictions/residual_plot.png", dpi=150)
print("  Residual plot saved → predictions/residual_plot.png")

#Save Model

joblib.dump(best_model, CONFIG["model_path"])
joblib.dump(scaler,     CONFIG["scaler_path"])
print(f"  Saved → {CONFIG['model_path']}")
print(f"  Saved → {CONFIG['scaler_path']}")