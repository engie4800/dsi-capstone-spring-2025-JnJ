import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

df = pd.read_csv("feature_validation_kevin.csv")

# Step 1: Replace "N/A" strings with actual NaN values
df_clean = df.replace("N/A", np.nan)

# Step 2: Drop rows where either predicted or actual is NaN
df_clean = df_clean.dropna(subset=["Number of Patients", "Number of Patients (Actual Value)", "Number of RCTs", "Number of RCTs (Actual Value)"])

# Step 3: Convert to numeric (in case they’re strings)
df_clean["Number of Patients"] = pd.to_numeric(df_clean["Number of Patients"])
df_clean["Number of Patients (Actual Value)"] = pd.to_numeric(df_clean["Number of Patients (Actual Value)"])
df_clean["Number of RCTs"] = pd.to_numeric(df_clean["Number of RCTs"])
df_clean["Number of RCTs (Actual Value)"] = pd.to_numeric(df_clean["Number of RCTs (Actual Value)"])

# Step 4: Extract values
y_pred_patient = df_clean["Number of Patients"].values
y_true_patient = df_clean["Number of Patients (Actual Value)"].values
y_pred_rct = df_clean["Number of RCTs"].values
y_true_rct = df_clean["Number of RCTs (Actual Value)"].values


# Step 5: Compute metrics
mse_patient = mean_squared_error(y_true_patient, y_pred_patient)
rmse_patient = mse_patient ** 0.5
r2_patient = r2_score(y_true_patient, y_pred_patient)

mse_rct = mean_squared_error(y_true_rct, y_pred_rct)
rmse_rct = mse_rct ** 0.5
r2_rct = r2_score(y_true_rct, y_pred_rct)

# Step 6: Print results
print(f"RMSE of patient: {rmse_patient:.4f}")
print(f"R² of patient: {r2_patient:.4f}")

print(f"RMSE of RCT: {rmse_rct:.4f}")
print(f"R² of RCT: {r2_rct:.4f}")