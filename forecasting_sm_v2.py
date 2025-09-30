import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# ==========================
# 1. READ DATA
# ==========================
file_path = "Harmon Foods Data.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# Add artificial 'Exceptional Event' column
df['Exceptional Event'] = 0  # Initialize all values to 0
df.loc[12, 'Exceptional Event'] = 1  # Row 13 (0-based index 12)
df.loc[38, 'Exceptional Event'] = 1  # Row 39 (0-based index 38)
df.loc[40, 'Exceptional Event'] = 1  # Row 41 (0-based index 40)
df.loc[41, 'Exceptional Event'] = 1  # Row 42 (0-based index 41)
df.loc[44, 'Exceptional Event'] = 1  # Row 45 (0-based index 44)

# print(df.iloc[40])  # Print to verify

# Convert all relevant columns to float immediately after loading
cols_to_float = ['Sales', 'CP', 'CP(t-1)', 'CP(t-2)', 'DA', 'DA(t-1)', 'DA(t-2)', 'SeasIndx', 'Exceptional Event']
df[cols_to_float] = df[cols_to_float].astype(float)

    
# Apply log transformations to DA variables
df['DA'] = np.log1p(df['DA'])  # Log-transform DA to reduce skewness
df['DA(t-1)'] = np.log1p(df['DA(t-1)'])  # Log-transform DA to reduce skewness
df['DA(t-2)'] = np.log1p(df['DA(t-2)'])  # Log-transform DA to reduce skewness

df['CP'] = np.log1p(df['CP'])  # Log-transform CP to reduce skewness
df['CP(t-1)'] = np.log1p(df['CP(t-1)'])  # Log-transform CP to reduce skewness
df['CP(t-2)'] = np.log1p(df['CP(t-2)'])  # Log-transform CP to reduce skewness

# Create lag features
for lag in [12]:
    df[f"lag_{lag}"] = df["SeasIndx"].shift(lag)

# Drop rows with NaN (first few because of shift)
# df = df.dropna()

# ==========================
# 2. PREPARE VARIABLES
# ==========================
sales_v1_cols = ['CP(t-1)', 'CP(t-2)', 'DA(t-1)', 'DA(t-2)', 'SeasIndx']  # Including SeasIndx as it is 100% predictable (being a deterministic function)


# ---- v2.1: include current CP and DA as predictors ----
# sales_v2_cols = ['CP', 'CP(t-1)', 'CP(t-2)', 'DA', 'DA(t-1)', 'DA(t-2)', 'SeasIndx']

# # ---- v2.2: exclude CP(t-2) and DA(t-1), DA(t-2) (based on highest p-values) ----
sales_v2_cols = ['CP', 'DA', 'SeasIndx']

# cp_cols = ['CP(t-1)', 'CP(t-2)', 'SeasIndx']
# da_cols = ['DA(t-1)', 'DA(t-2)', 'SeasIndx', 'Exceptional Event']
# si_cols = ['lag_4', 'lag_5', 'lag_7', 'lag_12']

si_cols = ['lag_12']
cp_cols = ['CP(t-1)', 'SeasIndx']
da_cols = ['DA(t-1)', 'DA(t-2)', 'Exceptional Event']

sales_target = 'Sales'
cp_target = 'CP'
da_target = 'DA'
si_target = 'SeasIndx'

# ==========================
# 3. ROLLING WINDOW SETTINGS
# ==========================
initial_train = 24  # start with first 36 months as training
n = len(df)

sales_pred_v1 = []
sales_pred_v2 = []
sales_true = []

CP_pred = []
DA_pred = []
SI_pred = []
CP_true = []
DA_true = []
SI_true = []

# To save coefficient stats
coef_stats_si = []
coef_stats_cp = []
coef_stats_da = []
coef_stats_v1 = []
coef_stats_v2 = []

# To save R² values for each model
r2_cp = []
r2_da = []
r2_si = []
r2_sales_v1 = []
r2_sales_v2 = []

# Lists to store instantaneous errors per iteration
instantaneous_rmse_CP = []
instantaneous_rmse_DA = []
instantaneous_rmse_SI = []
instantaneous_rmse_sales_v1 = []
instantaneous_rmse_sales_v2 = []

instantaneous_mape_CP = []
instantaneous_mape_DA = []
instantaneous_mape_SI = []
instantaneous_mape_sales_v1 = []
instantaneous_mape_sales_v2 = []

# Lists to store cumulative RMSE and MAPE for ensemble weighting
cumulative_rmse_sales_v1 = []
cumulative_rmse_sales_v2 = []
cumulative_mape_sales_v1 = []
cumulative_mape_sales_v2 = []

# Lists to store January 1988 predictions for each iteration
jan88_pred_sales_v1 = []
jan88_pred_sales_v2 = []

iteration_numbers = []

# ==========================
# 4. ROLLING FORECAST LOOP
# ==========================
for t in tqdm(range(initial_train, n), desc="Rolling Forecast"):
    # train = df.iloc[t-initial_train+12:t].copy()
    # test  = df.iloc[t:t+1].copy()
    train = df.iloc[12:t].copy()
    test  = df.iloc[t:t+1].copy()


    # ==========================
    # 4a. Predict CP_t and DA_t (using OLS on their lags) and SeasIndx (using OLS on its previous data)
    # ==========================
    
    X_si = sm.add_constant(train[si_cols], has_constant='add')
    si_model = sm.OLS(train[si_target], X_si).fit()
    r2_si.append(si_model.rsquared)
    X_si_test = sm.add_constant(test[si_cols], has_constant='add')
    si_pred = si_model.predict(X_si_test).iloc[0]

    coef_stats_si.append(si_model.summary2(alpha=0.20).tables[1])

    # ---- If including SI in the CP and DA models, need to use the predicted value ----
    X_cp = sm.add_constant(train[cp_cols], has_constant='add')
    cp_model = sm.OLS(train[cp_target], X_cp).fit()
    r2_cp.append(cp_model.rsquared)
    X_cp_test = sm.add_constant(test[cp_cols], has_constant='add')
    X_cp_test.loc[:, 'SeasIndx'] = si_pred
    cp_pred = cp_model.predict(X_cp_test).iloc[0]
    
    coef_stats_cp.append(cp_model.summary2(alpha=0.20).tables[1])

    X_da = sm.add_constant(train[da_cols], has_constant='add')
    da_model = sm.OLS(train[da_target], X_da).fit()
    r2_da.append(da_model.rsquared)
    X_da_test = sm.add_constant(test[da_cols], has_constant='add')
    # X_da_test.loc[:, 'SeasIndx'] = si_pred
    X_da_test.loc[:, 'Exceptional Event'] = test['Exceptional Event'].values[0]
    da_pred = da_model.predict(X_da_test).iloc[0]
    
    coef_stats_da.append(da_model.summary2(alpha=0.20).tables[1])

    # ==========================
    # 4b. Sales Model v1: lag-only
    # ==========================
    X1_train = sm.add_constant(train[sales_v1_cols], has_constant='add')
    y1_train = train[sales_target]
    X1_test = sm.add_constant(test[sales_v1_cols], has_constant='add')
    X1_test.loc[:, 'SeasIndx'] = si_pred
    
    sales_model_v1 = sm.OLS(y1_train, X1_train).fit()
    r2_sales_v1.append(sales_model_v1.rsquared)
    y1_pred = sales_model_v1.predict(X1_test).iloc[0]
    
    coef_stats_v1.append(sales_model_v1.summary2(alpha=0.20).tables[1])
    
    # ==========================
    # 4c. Sales Model v2: full + predicted CP/DA
    # ==========================
    # Use observed CP/DA in training
    X2_train = sm.add_constant(train[sales_v2_cols], has_constant='add')
    y2_train = train[sales_target]
    sales_model_v2 = sm.OLS(y2_train, X2_train).fit()
    r2_sales_v2.append(sales_model_v2.rsquared)
    
    coef_stats_v2.append(sales_model_v2.summary2(alpha=0.20).tables[1])
    
    # Test row: replace CP/DA/SeasIndx with predicted
    X2_test = test[sales_v2_cols].copy().astype(float)
    X2_test.loc[:, 'CP'] = cp_pred
    X2_test.loc[:, 'DA'] = da_pred
    X2_test.loc[:, 'SeasIndx'] = si_pred
    X2_test = sm.add_constant(X2_test, has_constant='add')
    
    y2_pred = sales_model_v2.predict(X2_test).iloc[0]
    
    # ==========================
    # Save results
    # ==========================
    sales_pred_v1.append(y1_pred)
    sales_pred_v2.append(y2_pred)
    sales_true.append(test[sales_target].values[0])
    
    CP_pred.append(cp_pred)
    DA_pred.append(da_pred)
    SI_pred.append(si_pred)
    CP_true.append(test[cp_target].values[0])
    DA_true.append(test[da_target].values[0])
    SI_true.append(test[si_target].values[0])
    
    # Calculate instantaneous errors for this iteration
    # RMSE (single value)
    instantaneous_rmse_CP.append(abs(test[cp_target].values[0] - cp_pred))
    instantaneous_rmse_DA.append(abs(test[da_target].values[0] - da_pred))
    instantaneous_rmse_SI.append(abs(test[si_target].values[0] - si_pred))
    instantaneous_rmse_sales_v1.append(abs(test[sales_target].values[0] - y1_pred))
    instantaneous_rmse_sales_v2.append(abs(test[sales_target].values[0] - y2_pred))
    
    # MAPE (single value) - with zero protection
    cp_true_val = test[cp_target].values[0]
    da_true_val = test[da_target].values[0]
    si_true_val = test[si_target].values[0]
    sales_true_val = test[sales_target].values[0]
    
    instantaneous_mape_CP.append(abs(cp_true_val - cp_pred) / cp_true_val * 100 if cp_true_val != 0 else 0.0)
    instantaneous_mape_DA.append(abs(da_true_val - da_pred) / da_true_val * 100 if da_true_val != 0 else 0.0)
    instantaneous_mape_SI.append(abs(si_true_val - si_pred) / si_true_val * 100 if si_true_val != 0 else 0.0)
    instantaneous_mape_sales_v1.append(abs(sales_true_val - y1_pred) / sales_true_val * 100 if sales_true_val != 0 else 0.0)
    instantaneous_mape_sales_v2.append(abs(sales_true_val - y2_pred) / sales_true_val * 100 if sales_true_val != 0 else 0.0)
    
    # Store iteration number
    iteration_numbers.append(t)
    
    # ==========================
    # 4d. January 1988 Prediction for this iteration (ensemble tracking)
    # ==========================
    
    # Create January 1988 row for prediction
    jan_88_row = {
        'CP': np.nan,
        'CP(t-1)': df['CP'].iloc[-1],        # December 1987 CP (log-transformed)
        'CP(t-2)': df['CP(t-1)'].iloc[-1],   # November 1987 CP (log-transformed)
        'DA': np.nan,
        'DA(t-1)': df['DA'].iloc[-1],        # December 1987 DA (log-transformed)
        'DA(t-2)': df['DA(t-1)'].iloc[-1],   # November 1987 DA (log-transformed)
        'SeasIndx': df['SeasIndx'].iloc[-1], # Use last known SeasIndx as starting point
        'lag_12': df['SeasIndx'].iloc[-12],  # lag_12 is 12 months back from Dec 1987
        'Exceptional Event': 0,
        'Sales': np.nan
    }
    
    # Create temporary dataframe for Jan88 prediction
    df_temp_jan88 = pd.concat([df, pd.DataFrame([jan_88_row])], ignore_index=True)
    jan88_idx = len(df_temp_jan88) - 1
    
    # Step 1: Predict SeasIndx for January 1988
    X_si_jan88 = sm.add_constant(df_temp_jan88.loc[[jan88_idx], si_cols], has_constant='add')
    si_pred_jan88 = si_model.predict(X_si_jan88).iloc[0]
    df_temp_jan88.at[jan88_idx, 'SeasIndx'] = si_pred_jan88
    
    # Step 2: Predict CP for January 1988
    X_cp_jan88 = sm.add_constant(df_temp_jan88.loc[[jan88_idx], cp_cols], has_constant='add')
    cp_pred_jan88 = cp_model.predict(X_cp_jan88).iloc[0]
    df_temp_jan88.at[jan88_idx, 'CP'] = cp_pred_jan88
    
    # Step 3: Predict DA for January 1988
    X_da_jan88 = sm.add_constant(df_temp_jan88.loc[[jan88_idx], da_cols], has_constant='add')
    da_pred_jan88 = da_model.predict(X_da_jan88).iloc[0]
    df_temp_jan88.at[jan88_idx, 'DA'] = da_pred_jan88
    
    # Step 4: Predict Sales V1 (lag-only model) for January 1988
    X_sales_v1_jan88 = sm.add_constant(df_temp_jan88.loc[[jan88_idx], sales_v1_cols], has_constant='add')
    sales_pred_v1_jan88_iter = sales_model_v1.predict(X_sales_v1_jan88).iloc[0]
    
    # Step 5: Predict Sales V2 (full model with predicted CP/DA) for January 1988
    X_sales_v2_jan88 = df_temp_jan88.loc[[jan88_idx], sales_v2_cols].copy()
    X_sales_v2_jan88['CP'] = cp_pred_jan88
    X_sales_v2_jan88['DA'] = da_pred_jan88
    X_sales_v2_jan88['SeasIndx'] = si_pred_jan88
    X_sales_v2_jan88 = sm.add_constant(X_sales_v2_jan88, has_constant='add')
    sales_pred_v2_jan88_iter = sales_model_v2.predict(X_sales_v2_jan88).iloc[0]
    
    # Store January 1988 predictions for this iteration
    jan88_pred_sales_v1.append(sales_pred_v1_jan88_iter)
    jan88_pred_sales_v2.append(sales_pred_v2_jan88_iter)
    
    # ==========================
    # 4e. Compute cumulative RMSE and MAPE for ensemble weighting
    # ==========================
    
    # Convert current predictions to arrays for calculation
    sales_true_array = np.array(sales_true)
    sales_pred_v1_array = np.array(sales_pred_v1)
    sales_pred_v2_array = np.array(sales_pred_v2)
    
    # Calculate cumulative RMSE
    if len(sales_true_array) > 0:
        cumulative_rmse_v1 = np.sqrt(np.mean((sales_true_array - sales_pred_v1_array)**2))
        cumulative_rmse_v2 = np.sqrt(np.mean((sales_true_array - sales_pred_v2_array)**2))
        
        # Calculate cumulative MAPE with zero protection
        mask_v1 = sales_true_array != 0
        mask_v2 = sales_true_array != 0
        
        if np.any(mask_v1):
            cumulative_mape_v1 = np.mean(np.abs((sales_true_array[mask_v1] - sales_pred_v1_array[mask_v1]) / sales_true_array[mask_v1])) * 100
        else:
            cumulative_mape_v1 = 0.0
            
        if np.any(mask_v2):
            cumulative_mape_v2 = np.mean(np.abs((sales_true_array[mask_v2] - sales_pred_v2_array[mask_v2]) / sales_true_array[mask_v2])) * 100
        else:
            cumulative_mape_v2 = 0.0
    else:
        cumulative_rmse_v1 = 0.0
        cumulative_rmse_v2 = 0.0
        cumulative_mape_v1 = 0.0
        cumulative_mape_v2 = 0.0
    
    # Store cumulative metrics
    cumulative_rmse_sales_v1.append(cumulative_rmse_v1)
    cumulative_rmse_sales_v2.append(cumulative_rmse_v2)
    cumulative_mape_sales_v1.append(cumulative_mape_v1)
    cumulative_mape_sales_v2.append(cumulative_mape_v2)
    

# ==========================
# 5. METRICS
# ==========================

# Convert lists to numpy arrays for calculations

sales_true = np.array(sales_true)
sales_pred_v1 = np.array(sales_pred_v1)
sales_pred_v2 = np.array(sales_pred_v2)
DA_true = np.array(DA_true)
DA_pred = np.array(DA_pred)
CP_true = np.array(CP_true)
CP_pred = np.array(CP_pred)
SI_true = np.array(SI_true)
SI_pred = np.array(SI_pred)

def mape(y_true, y_pred):
    # Create mask to exclude zero values to avoid division by zero
    mask = y_true != 0
    if not np.any(mask):  # All values are zero
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

results = pd.DataFrame({
    "Actual": sales_true,
    "Pred_v1": sales_pred_v1,
    "Pred_v2": sales_pred_v2,
    "Actual_CP": CP_true,
    "Pred_CP": CP_pred,
    "Actual_DA": DA_true,
    "Pred_DA": DA_pred,
    "Actual_SI": SI_true,
    "Pred_SI": SI_pred
})

# ---- Prediction Error Metrics for CP, DA, SI ----

rmse_DA = np.sqrt(np.mean((DA_true - DA_pred)**2))
mape_DA = mape(DA_true, DA_pred)

rmse_CP = np.sqrt(np.mean((CP_true - CP_pred)**2))
mape_CP = mape(CP_true, CP_pred)

rmse_SI = np.sqrt(np.mean((SI_true - SI_pred)**2))
mape_SI = mape(SI_true, SI_pred)


# ---- Sales Error Metrics ----
rmse_v1 = np.sqrt(np.mean((sales_true - sales_pred_v1)**2))
rmse_v2 = np.sqrt(np.mean((sales_true - sales_pred_v2)**2))
mape_v1 = mape(sales_true, sales_pred_v1)
mape_v2 = mape(sales_true, sales_pred_v2)

print("===== Rolling Forecast Results =====")
print(results)

print("\nModel v1 (Lag Only)")
print("==== Sales Prediction Errors ====")
print(f"RMSE : {rmse_v1:,.2f}")
print(f"MAPE : {mape_v1:.2f}%")

print("\nModel v2 (Full + Predicted CP/DA)")
print(f"RMSE : {rmse_v2:,.2f}")
print(f"MAPE : {mape_v2:.2f}%")

print("\n==== Predictors Errors CP, DA, SI ====")

print(f"CP RMSE : {rmse_CP:,.2f}")
print(f"CP MAPE : {mape_CP:.2f}%")

print(f"DA RMSE : {rmse_DA:,.2f}")
print(f"DA MAPE : {mape_DA:.2f}%")

print(f"SI RMSE : {rmse_SI:,.2f}")
print(f"SI MAPE : {mape_SI:.2f}%")

# ==========================
# R² Statistics
# ==========================
print("\n==== R² Statistics ====")

# Convert R² lists to numpy arrays for calculations
r2_cp_arr = np.array(r2_cp)
r2_da_arr = np.array(r2_da)
r2_si_arr = np.array(r2_si)
r2_sales_v1_arr = np.array(r2_sales_v1)
r2_sales_v2_arr = np.array(r2_sales_v2)

print(f"CP Model - Mean R²: {np.mean(r2_cp_arr):.4f}, Std: {np.std(r2_cp_arr):.4f}, Last: {r2_cp_arr[-1]:.4f}")
print(f"DA Model - Mean R²: {np.mean(r2_da_arr):.4f}, Std: {np.std(r2_da_arr):.4f}, Last: {r2_da_arr[-1]:.4f}")
print(f"SI Model - Mean R²: {np.mean(r2_si_arr):.4f}, Std: {np.std(r2_si_arr):.4f}, Last: {r2_si_arr[-1]:.4f}")
print(f"Sales V1 - Mean R²: {np.mean(r2_sales_v1_arr):.4f}, Std: {np.std(r2_sales_v1_arr):.4f}, Last: {r2_sales_v1_arr[-1]:.4f}")
print(f"Sales V2 - Mean R²: {np.mean(r2_sales_v2_arr):.4f}, Std: {np.std(r2_sales_v2_arr):.4f}, Last: {r2_sales_v2_arr[-1]:.4f}")

# ==========================
# 6. Example: Inspect t-stats and p-values for last iteration
# ==========================
print("\nLast SeasIndx model coefficients:\n", coef_stats_si[-1])
print("\nLast CP model coefficients:\n", coef_stats_cp[-1])
print("\nLast DA model coefficients:\n", coef_stats_da[-1])
print("\nLast v1 model coefficients:\n", coef_stats_v1[-1])
print("\nLast v2 model coefficients:\n", coef_stats_v2[-1])

# ==========================
# 7. VISUALIZATION: Instantaneous Error Evolution
# ==========================

# Create simple plots showing instantaneous error evolution
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Instantaneous Error Evolution Across Iterations', fontsize=14, fontweight='bold')

# Plot 1: Instantaneous RMSE for Predictors
ax1.plot(iteration_numbers, instantaneous_rmse_CP, 'b-', linewidth=1.5, label='CP', marker='o', markersize=2)
ax1.plot(iteration_numbers, instantaneous_rmse_DA, 'r-', linewidth=1.5, label='DA', marker='s', markersize=2)
ax1.plot(iteration_numbers, instantaneous_rmse_SI, 'g-', linewidth=1.5, label='SI', marker='^', markersize=2)
ax1.set_title('Instantaneous RMSE - Predictors')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Absolute Error')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Instantaneous MAPE for Predictors
ax2.plot(iteration_numbers, instantaneous_mape_CP, 'b-', linewidth=1.5, label='CP', marker='o', markersize=2)
ax2.plot(iteration_numbers, instantaneous_mape_DA, 'r-', linewidth=1.5, label='DA', marker='s', markersize=2)
ax2.plot(iteration_numbers, instantaneous_mape_SI, 'g-', linewidth=1.5, label='SI', marker='^', markersize=2)
ax2.set_title('Instantaneous MAPE - Predictors')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('MAPE (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Instantaneous RMSE for Sales Models
ax3.plot(iteration_numbers, instantaneous_rmse_sales_v1, 'purple', linewidth=1.5, label='Sales V1', marker='d', markersize=2)
ax3.plot(iteration_numbers, instantaneous_rmse_sales_v2, 'orange', linewidth=1.5, label='Sales V2', marker='x', markersize=3)
ax3.set_title('Instantaneous RMSE - Sales Models')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Absolute Error')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Instantaneous MAPE for Sales Models
ax4.plot(iteration_numbers, instantaneous_mape_sales_v1, 'purple', linewidth=1.5, label='Sales V1', marker='d', markersize=2)
ax4.plot(iteration_numbers, instantaneous_mape_sales_v2, 'orange', linewidth=1.5, label='Sales V2', marker='x', markersize=3)
ax4.set_title('Instantaneous MAPE - Sales Models')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('MAPE (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/instantaneous_error_evolution.png', dpi=300, bbox_inches='tight')
print(f"\nInstantaneous error evolution plot saved as 'instantaneous_error_evolution.png'")
plt.show()

# ==========================
# 8. EXPLOIT: Add January 1988 row to df and compute forecasts
# ==========================

# Add a row for January 1988 to df, with placeholders for CP, DA, Sales, etc.
jan_88_row = {
  'CP': np.nan,
  'CP(t-1)': df['CP'].iloc[-1],        # December 1987 CP (log-transformed)
  'CP(t-2)': df['CP(t-1)'].iloc[-1],   # November 1987 CP (log-transformed)
  'DA': np.nan,
  'DA(t-1)': df['DA'].iloc[-1],        # December 1987 DA (log-transformed)
  'DA(t-2)': df['DA(t-1)'].iloc[-1],   # November 1987 DA (log-transformed)
  'SeasIndx': df['SeasIndx'].iloc[-1], # Use last known SeasIndx as starting point
  'lag_4': df['SeasIndx'].iloc[-4],    # lag_4 is 4 months back from Dec 1987
  'lag_5': df['SeasIndx'].iloc[-5],    # lag_5 is 5 months back from Dec 1987
  'lag_7': df['SeasIndx'].iloc[-7],    # lag_7 is 7 months back from Dec 1987
  'lag_12': df['SeasIndx'].iloc[-12],  # lag_12 is 12 months back from Dec 1987
  'Exceptional Event': 0,
  'Sales': np.nan
}

# Append the row to df
df_jan88 = pd.concat([df, pd.DataFrame([jan_88_row])], ignore_index=True)

# Use column selections defined earlier
jan88_idx = len(df_jan88) - 1

# Step 1: Predict SeasIndx for January 1988
X_si_jan88 = sm.add_constant(df_jan88.loc[[jan88_idx], si_cols], has_constant='add')
si_pred_jan88 = si_model.predict(X_si_jan88).iloc[0]
df_jan88.at[jan88_idx, 'SeasIndx'] = si_pred_jan88

# Step 2: Predict CP for January 1988
X_cp_jan88 = sm.add_constant(df_jan88.loc[[jan88_idx], cp_cols], has_constant='add')
cp_pred_jan88 = cp_model.predict(X_cp_jan88).iloc[0]
df_jan88.at[jan88_idx, 'CP'] = cp_pred_jan88

# Step 3: Predict DA for January 1988
X_da_jan88 = sm.add_constant(df_jan88.loc[[jan88_idx], da_cols], has_constant='add')
da_pred_jan88 = da_model.predict(X_da_jan88).iloc[0]
df_jan88.at[jan88_idx, 'DA'] = da_pred_jan88

# Step 4: Predict Sales V1 (lag-only model)
X_sales_v1_jan88 = sm.add_constant(df_jan88.loc[[jan88_idx], sales_v1_cols], has_constant='add')
sales_pred_v1_jan88 = sales_model_v1.predict(X_sales_v1_jan88).iloc[0]
df_jan88.at[jan88_idx, 'Sales'] = sales_pred_v1_jan88

# Step 5: Predict Sales V2 (full model with predicted CP/DA)
X_sales_v2_jan88 = df_jan88.loc[[jan88_idx], sales_v2_cols].copy()
X_sales_v2_jan88['CP'] = cp_pred_jan88
X_sales_v2_jan88['DA'] = da_pred_jan88
X_sales_v2_jan88['SeasIndx'] = si_pred_jan88
X_sales_v2_jan88 = sm.add_constant(X_sales_v2_jan88, has_constant='add')
sales_pred_v2_jan88 = sales_model_v2.predict(X_sales_v2_jan88).iloc[0]

forecasted_values = {
  "Forecasted_CP": np.expm1(cp_pred_jan88),
  "Forecasted_DA": np.expm1(da_pred_jan88),
  "Forecasted_SI": si_pred_jan88,
  "Forecasted_Sales_V1": sales_pred_v1_jan88,
  "Forecasted_Sales_V2": sales_pred_v2_jan88
}

print("\n===== January 1988 Forecasting =====")
for key, value in forecasted_values.items():
  print(f"{key}: {value:,.2f}")

# ==========================
# 10. WEIGHTED MEAN ENSEMBLE FOR JANUARY 1988
# ==========================

print("\n===== Weighted Mean Ensemble Analysis =====")

# Convert January 1988 predictions to arrays
jan88_pred_sales_v1_array = np.array(jan88_pred_sales_v1)
jan88_pred_sales_v2_array = np.array(jan88_pred_sales_v2)

# Get final cumulative RMSE for weighting (use last values)
final_rmse_v1 = cumulative_rmse_sales_v1[-1] if cumulative_rmse_sales_v1 else 1.0
final_rmse_v2 = cumulative_rmse_sales_v2[-1] if cumulative_rmse_sales_v2 else 1.0

# Prevent division by zero
if final_rmse_v1 == 0:
    final_rmse_v1 = 1e-8
if final_rmse_v2 == 0:
    final_rmse_v2 = 1e-8

# Calculate inverse RMSE weights
inverse_rmse_v1 = 1.0 / final_rmse_v1
inverse_rmse_v2 = 1.0 / final_rmse_v2

# Normalize weights to sum to 1
total_weight = inverse_rmse_v1 + inverse_rmse_v2
weight_v1 = inverse_rmse_v1 / total_weight
weight_v2 = inverse_rmse_v2 / total_weight

print(f"Final Cumulative RMSE - V1: {final_rmse_v1:,.4f}, V2: {final_rmse_v2:,.4f}")
print(f"Inverse RMSE Weights - V1: {weight_v1:.4f}, V2: {weight_v2:.4f}")

# Calculate weighted ensemble for all January 1988 predictions across iterations
jan88_ensemble_predictions = weight_v1 * jan88_pred_sales_v1_array + weight_v2 * jan88_pred_sales_v2_array

# Final ensemble prediction (last iteration)
final_jan88_ensemble = jan88_ensemble_predictions[-1]

print(f"\nJanuary 1988 Predictions Evolution:")
print(f"Final Sales V1 Prediction: {jan88_pred_sales_v1_array[-1]:,.2f}")
print(f"Final Sales V2 Prediction: {jan88_pred_sales_v2_array[-1]:,.2f}")
print(f"Final Weighted Ensemble Prediction: {final_jan88_ensemble:,.2f}")

# Calculate statistics across all ensemble predictions
ensemble_mean = np.mean(jan88_ensemble_predictions)
ensemble_std = np.std(jan88_ensemble_predictions)
ensemble_min = np.min(jan88_ensemble_predictions)
ensemble_max = np.max(jan88_ensemble_predictions)

print(f"\nEnsemble Prediction Statistics Across Iterations:")
print(f"Mean: {ensemble_mean:,.2f}")
print(f"Std Dev: {ensemble_std:,.2f}")
print(f"Min: {ensemble_min:,.2f}")
print(f"Max: {ensemble_max:,.2f}")

# Add ensemble to forecasted values
forecasted_values["Forecasted_Sales_Ensemble"] = final_jan88_ensemble

# ==========================
# 10b. VISUALIZATION: Ensemble and Cumulative Metrics
# ==========================

# Create additional plots for ensemble analysis
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Ensemble Analysis and Cumulative Metrics', fontsize=14, fontweight='bold')

# Plot 1: January 1988 Predictions Evolution
ax1.plot(iteration_numbers, jan88_pred_sales_v1, 'purple', linewidth=2, label='Sales V1', marker='o', markersize=3)
ax1.plot(iteration_numbers, jan88_pred_sales_v2, 'orange', linewidth=2, label='Sales V2', marker='s', markersize=3)
ax1.plot(iteration_numbers, jan88_ensemble_predictions, 'red', linewidth=2.5, label='Weighted Ensemble', marker='^', markersize=3)
ax1.set_title('January 1988 Sales Predictions Evolution')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Predicted Sales')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative RMSE Evolution
ax2.plot(iteration_numbers, cumulative_rmse_sales_v1, 'purple', linewidth=2, label='Sales V1', marker='o', markersize=3)
ax2.plot(iteration_numbers, cumulative_rmse_sales_v2, 'orange', linewidth=2, label='Sales V2', marker='s', markersize=3)
ax2.set_title('Cumulative RMSE Evolution')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cumulative RMSE')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cumulative MAPE Evolution
ax3.plot(iteration_numbers, cumulative_mape_sales_v1, 'purple', linewidth=2, label='Sales V1', marker='o', markersize=3)
ax3.plot(iteration_numbers, cumulative_mape_sales_v2, 'orange', linewidth=2, label='Sales V2', marker='s', markersize=3)
ax3.set_title('Cumulative MAPE Evolution')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Cumulative MAPE (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Ensemble Weights Evolution
weights_v1_evolution = []
weights_v2_evolution = []

for i in range(len(cumulative_rmse_sales_v1)):
    rmse_v1 = cumulative_rmse_sales_v1[i] if cumulative_rmse_sales_v1[i] > 0 else 1e-8
    rmse_v2 = cumulative_rmse_sales_v2[i] if cumulative_rmse_sales_v2[i] > 0 else 1e-8
    
    inv_rmse_v1 = 1.0 / rmse_v1
    inv_rmse_v2 = 1.0 / rmse_v2
    total_weight = inv_rmse_v1 + inv_rmse_v2
    
    weights_v1_evolution.append(inv_rmse_v1 / total_weight)
    weights_v2_evolution.append(inv_rmse_v2 / total_weight)

ax4.plot(iteration_numbers, weights_v1_evolution, 'purple', linewidth=2, label='V1 Weight', marker='o', markersize=3)
ax4.plot(iteration_numbers, weights_v2_evolution, 'orange', linewidth=2, label='V2 Weight', marker='s', markersize=3)
ax4.set_title('Ensemble Weights Evolution (Inverse RMSE)')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Weight')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/ensemble_analysis.png', dpi=300, bbox_inches='tight')
print(f"Ensemble analysis plot saved as 'ensemble_analysis.png'")
plt.show()

# ==========================
# 9. COMPARE: Use and compare the Forecasted CP, DA, SI and SALES with the Company forecasts for January 1988
# ==========================

print("\n===== Comparison with Company Forecasts =====")

official_values = {
  "Official_CP": 100000.00,
  "Official_DA": 500000.00,
}

print(f"Official CP forecast: {official_values['Official_CP']:,.2f}")
print(f"Official DA forecast: {official_values['Official_DA']:,.2f}")

# Convert official values to log scale (to match our transformed data)
official_cp_log = np.log1p(official_values["Official_CP"])
official_da_log = np.log1p(official_values["Official_DA"])

print(f"Official CP (log-transformed): {official_cp_log:.4f}")
print(f"Official DA (log-transformed): {official_da_log:.4f}")

# Step 1: Predict Sales V2 using official CP and DA values
X_sales_v2_official = df_jan88.loc[[jan88_idx], sales_v2_cols].copy()
X_sales_v2_official['CP'] = official_cp_log
X_sales_v2_official['DA'] = official_da_log
X_sales_v2_official['SeasIndx'] = si_pred_jan88
X_sales_v2_official = sm.add_constant(X_sales_v2_official, has_constant='add')
sales_official_pred_v2 = sales_model_v2.predict(X_sales_v2_official).iloc[0]

print(f"\nSales prediction (Model V2) using predicted CP/DA: {sales_pred_v2_jan88:,.2f}")
print(f"Sales prediction (Model V2) using official CP/DA: {sales_official_pred_v2:,.2f}")

# Step 2: Predict Sales V1 using lag values (this shouldn't change much since V1 doesn't use current CP/DA)
sales_official_pred_v1 = sales_pred_v1_jan88  # Same as before since V1 uses only lags

print(f"Sales prediction (Model V1) using lags: {sales_official_pred_v1:,.2f}")

# Step 3: Compare differences
print(f"\n===== Comparison Summary =====")
print(f"CP Forecast - Our Model: {np.expm1(cp_pred_jan88):,.2f} vs Official: {official_values['Official_CP']:,.2f}")
print(f"DA Forecast - Our Model: {np.expm1(da_pred_jan88):,.2f} vs Official: {official_values['Official_DA']:,.2f}")
print(f"Sales V1 Forecast: {sales_pred_v1_jan88:,.2f}")
print(f"Sales V2 Forecast - Our CP/DA: {sales_pred_v2_jan88:,.2f}")
print(f"Sales Ensemble Forecast: {final_jan88_ensemble:,.2f}")
print(f"Sales V2 Forecast - Official CP/DA: {sales_official_pred_v2:,.2f}")

# Calculate differences
cp_diff = np.expm1(cp_pred_jan88) - official_values['Official_CP']
da_diff = np.expm1(da_pred_jan88) - official_values['Official_DA']
sales_diff_v2 = sales_pred_v2_jan88 - sales_official_pred_v2
sales_diff_ensemble = final_jan88_ensemble - sales_official_pred_v2

print(f"\n===== Differences (Our Model - Official) =====")
print(f"CP Difference: {cp_diff:,.2f} ({cp_diff/official_values['Official_CP']*100:+.1f}%)")
print(f"DA Difference: {da_diff:,.2f} ({da_diff/official_values['Official_DA']*100:+.1f}%)")
print(f"Sales V2 Difference: {sales_diff_v2:,.2f} ({sales_diff_v2/sales_official_pred_v2*100:+.1f}%)")
print(f"Sales Ensemble Difference: {sales_diff_ensemble:,.2f} ({sales_diff_ensemble/sales_official_pred_v2*100:+.1f}%)")
