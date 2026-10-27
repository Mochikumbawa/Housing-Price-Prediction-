"""
Housing Price Prediction (reconstructed)

NOTE: This is a reconstruction based on the shape of the saved predictions,
not the original code. Fill in the TODOs with your real file names and columns.

Clues from Recode-2by2.csv that shaped the design:
  - Predictions are positive, right-skewed, and bounded (min ~0.66M, max ~19.5M)
  - Different properties often get the *exact same* prediction -> tree-based model
  - The same property gets the *same* prediction on different contractDates
    -> contractDate was NOT a model feature; it is only carried into the output
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor  # swap for GradientBoostingRegressor / XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ---- TODO: set these to match your data ------------------------------------
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"            # must contain property_key and contractDate
TARGET = "price"                  # your sale price column
ID_COL = "property_key"
DATE_COL = "contractDate"
# -----------------------------------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Date is kept for the output file only, not used for training
features = [c for c in train.columns if c not in {TARGET, ID_COL, DATE_COL}]
num_cols = train[features].select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in features if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
])

# Prices are right-skewed, so train on the log scale
y = np.log1p(train[TARGET])
X_tr, X_val, y_tr, y_val = train_test_split(train[features], y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)

val_pred = np.expm1(model.predict(X_val))
val_true = np.expm1(y_val)
print(f"MAE : {mean_absolute_error(val_true, val_pred):,.0f}")
print(f"RMSE: {np.sqrt(mean_squared_error(val_true, val_pred)):,.0f}")
print(f"R2  : {r2_score(val_true, val_pred):.3f}")

# Refit on all training data, then predict the test set
model.fit(train[features], y)
out = test[[ID_COL, DATE_COL]].copy()
out["prediction"] = np.expm1(model.predict(test[features]))
out.to_csv("predictions.csv", index=False)
print("Saved predictions.csv", out.shape)
