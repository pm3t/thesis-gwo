import pandas as pd
import numpy as np
import math

# Read Result.csv
df = pd.read_csv(r"c:\GWO\Result.csv")

y_true = df['Actual'].values
y_ma = df['MA'].values
y_es = df['ES'].values
y_lr = df['LR'].values
y_eq = df['Equal_Average'].values
y_gwo = df['GWO_Ensemble'].values

# Absolute Percentage Error (APE)
ape_ma = np.abs((y_true - y_ma) / y_true)
ape_gwo = np.abs((y_true - y_gwo) / y_true)
ape_eq = np.abs((y_true - y_eq) / y_true)

# Difference in APE: d_i = APE_MA - APE_GWO
d_ape = ape_ma - ape_gwo
n = len(d_ape)
mean_d = np.mean(d_ape)
std_d = np.std(d_ape, ddof=1)
se_d = std_d / math.sqrt(n)
t_stat = mean_d / se_d

print("=== PURE NUMPY STATISTICAL SIGNIFICANCE TESTS ===")
print(f"Sample size N = {n}")
print(f"Mean MAPE MA = {np.mean(ape_ma)*100:.4f}%")
print(f"Mean MAPE GWO = {np.mean(ape_gwo)*100:.4f}%")
print(f"Mean MAPE Difference = {mean_d*100:.4f}%")
print(f"Std Dev of Difference = {std_d*100:.4f}%")
print(f"Standard Error of Mean Difference = {se_d*100:.6f}%")
print(f"Paired t-statistic = {t_stat:.4f}")

# 95% Confidence Interval for Mean Difference (z=1.96 for N=335)
z_crit = 1.95996
ci_low = (mean_d - z_crit * se_d) * 100
ci_high = (mean_d + z_crit * se_d) * 100
print(f"95% Confidence Interval for MAPE Difference: [{ci_low:.4f}%, {ci_high:.4f}%]")

# Diebold-Mariano Test for MAPE
# Loss differential d_t = e_ma^2 - e_gwo^2 (MSE) or |e_ma| - |e_gwo| (MAE) or APE_ma - APE_gwo (MAPE)
for name, e1, e2 in [
    ("MAPE", ape_ma, ape_gwo),
    ("MAE", np.abs(y_true - y_ma), np.abs(y_true - y_gwo)),
    ("MSE", (y_true - y_ma)**2, (y_true - y_gwo)**2)
]:
    diff = e1 - e2
    m_diff = np.mean(diff)
    s_diff = np.std(diff, ddof=1)
    se_diff = s_diff / math.sqrt(n)
    dm_stat = m_diff / se_diff
    print(f"\nDiebold-Mariano Test ({name}):")
    print(f"  Mean Loss Differential: {m_diff:.6f}")
    print(f"  DM-statistic: {dm_stat:.4f}")

# Also compare Equal Average vs GWO Ensemble
d_eq = ape_eq - ape_gwo
m_eq = np.mean(d_eq)
s_eq = np.std(d_eq, ddof=1)
se_eq = s_eq / math.sqrt(n)
t_stat_eq = m_eq / se_eq

print("\nEqual Average vs GWO Ensemble (MAPE):")
print(f"  Mean MAPE Equal Avg = {np.mean(ape_eq)*100:.4f}%")
print(f"  Mean MAPE GWO = {np.mean(ape_gwo)*100:.4f}%")
print(f"  Difference = {m_eq*100:.4f}%")
print(f"  Paired t-statistic = {t_stat_eq:.4f}")
