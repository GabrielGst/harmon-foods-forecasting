import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

# ==========================
# 1. READ DATA
# ==========================
file_path = "Harmon Foods Data.xlsx"
df = pd.read_excel(file_path)

# Ensure columns have consistent names (no hidden spaces)
df.columns = df.columns.str.strip()

# ==========================
# 2. PREPARE VARIABLES
# ==========================
# --- Predictors for Sales ---
# v1: only lagged CP/DA and seasonal index
sales_v1_cols = ['CP(t-1)', 'CP(t-2)', 'DA(t-1)', 'DA(t-2)', 'SeasIndx']

# v2: all variables (current CP/DA included)
sales_v2_cols = ['CP', 'CP(t-1)', 'CP(t-2)', 'DA', 'DA(t-1)', 'DA(t-2)', 'SeasIndx']

# --- Predictors for CP and DA ---
cp_cols = ['CP(t-1)', 'CP(t-2)']
da_cols = ['DA(t-1)', 'DA(t-2)']

# Target
sales_target = 'Sales'
cp_target = 'CP'
da_target = 'DA'

# ==========================
# 3. ROLLING WINDOW SETTINGS
# ==========================
initial_train = 24     # start with first 3 months as training
n = len(df)

# To collect forecasts
sales_pred_v1 = []
sales_pred_v2 = []
sales_true = []

# ==========================
# 4. ROLLING FORECAST LOOP
# ==========================

for t in tqdm(range(initial_train, n), desc="Rolling Forecast"):
    # --------------------------
    # TRAIN SET = rows [0:t)
    # TEST SET  = row [t] (next month)
    # --------------------------
    train = df.iloc[:t].copy()
    test  = df.iloc[t:t+1].copy()
    
    # Cast to float to safely store float predictions
    train[['Sales', 'CP','CP(t-1)','CP(t-2)','DA','DA(t-1)','DA(t-2)']] = train[['Sales', 'CP','CP(t-1)','CP(t-2)','DA','DA(t-1)','DA(t-2)']].astype(float)
    test[['Sales', 'CP','CP(t-1)','CP(t-2)','DA','DA(t-1)','DA(t-2)']] = test[['Sales', 'CP','CP(t-1)','CP(t-2)','DA','DA(t-1)','DA(t-2)']].astype(float)

    # ---- Model v1 (lag-only) ----
    X1_train = train[sales_v1_cols]
    y_train = train[sales_target]
    X1_test  = test[sales_v1_cols]

    model_v1 = LinearRegression().fit(X1_train, y_train)
    y1_pred  = model_v1.predict(X1_test)[0]

    # ---- Predict CP_t and DA_t for v2 ----
    Xcp_train = train[cp_cols]
    ycp_train = train[cp_target]
    Xcp_test  = test[cp_cols]

    Xda_train = train[da_cols]
    yda_train = train[da_target]
    Xda_test  = test[da_cols]

    cp_model = LinearRegression().fit(Xcp_train, ycp_train)
    da_model = LinearRegression().fit(Xda_train, yda_train)

    cp_pred = cp_model.predict(Xcp_test)[0]
    da_pred = da_model.predict(Xda_test)[0]

    # --------------------------
    # TRAIN SALES MODEL v2 (observed data)
    # --------------------------
    X2_train = train[sales_v2_cols]       # use actual CP/DA
    y2_train = train[sales_target]
    model_v2 = LinearRegression().fit(X2_train, y2_train)

    # --------------------------
    # PREPARE TEST DATA FOR PREDICTION (use predicted CP/DA)
    # --------------------------
    X2_test = test[sales_v2_cols].copy()
    X2_test = X2_test.astype(float)       # ensure floats
    X2_test.loc[:, 'CP'] = cp_pred        # replace only test row CP with prediction
    X2_test.loc[:, 'DA'] = da_pred        # replace only test row DA with prediction

    # PREDICT SALES
    y2_pred = model_v2.predict(X2_test)[0]

    # ---- Save results ----
    sales_pred_v1.append(y1_pred)
    sales_pred_v2.append(y2_pred)
    sales_true.append(test[sales_target].values[0])

# ==========================
# 5. METRICS
# ==========================
sales_true = np.array(sales_true)
sales_pred_v1 = np.array(sales_pred_v1)
sales_pred_v2 = np.array(sales_pred_v2)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

results = pd.DataFrame({
    "Actual": sales_true,
    "Pred_v1": sales_pred_v1,
    "Pred_v2": sales_pred_v2
})
  
rmse_v1 = np.sqrt(mean_squared_error(sales_true, sales_pred_v1))
rmse_v2 = np.sqrt(mean_squared_error(sales_true, sales_pred_v2))
mape_v1 = mape(sales_true, sales_pred_v1)
mape_v2 = mape(sales_true, sales_pred_v2)

print("===== Rolling Forecast Results =====")
print(results)
print("\nModel v1 (Lag Only)")
print(f"RMSE : {rmse_v1:,.2f}")
print(f"MAPE : {mape_v1:.2f}%")

print("\nModel v2 (Full + Predicted CP/DA)")
print(f"RMSE : {rmse_v2:,.2f}")
print(f"MAPE : {mape_v2:.2f}%")
