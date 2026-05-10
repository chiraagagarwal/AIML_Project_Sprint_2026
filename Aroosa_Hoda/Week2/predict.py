import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import joblib
from config import CONFIG

model  = joblib.load(CONFIG["model_path"])
scaler = joblib.load(CONFIG["scaler_path"])

# Raw inputs only, no feature engineering yet
raw_houses = pd.DataFrame({
    "MedInc":     [8.3, 3.5],
    "HouseAge":   [15,  40],
    "AveRooms":   [6.5, 4.2],
    "AveBedrms":  [1.1, 1.3],
    "Population": [800, 1200],
    "AveOccup":   [2.8, 3.1],
    "Latitude":   [37.8, 34.1],
    "Longitude":  [-122.4, -118.2],
})

# Feature engineering
raw_houses["log_median_income"]        = np.log1p(raw_houses["MedInc"])
raw_houses["rooms_per_household"]      = raw_houses["AveRooms"] / raw_houses["HouseAge"].clip(lower=1)
raw_houses["bedrooms_per_room"]        = raw_houses["AveBedrms"] / raw_houses["AveRooms"].clip(lower=1)
raw_houses["population_per_household"] = raw_houses["Population"] / raw_houses["AveOccup"].clip(lower=1)

# Select features
feature_cols = [
    "MedInc", "log_median_income", "HouseAge", "AveRooms",
    "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude",
    "rooms_per_household", "bedrooms_per_room", "population_per_household"
]
X = raw_houses[feature_cols]

# Scale → predict → invert log transform
X_scaled  = scaler.transform(X)
log_preds = model.predict(X_scaled)
prices    = np.expm1(log_preds) * 100_000

for i, price in enumerate(prices):
    print(f"House {i+1}: ${price:,.0f}")

pd.DataFrame({"house": range(1, len(prices)+1), "predicted_price": prices}) \
  .to_csv("house_price_pipeline/predictions/new_houses.csv", index=False)
print("Saved -> house_price_pipeline/predictions/new_houses.csv")