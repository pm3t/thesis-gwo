import pandas as pd
import numpy as np
from scipy import stats

# Read Result.csv
df = pd.read_csv(r"c:\GWO\Result.csv")

y_true = df['Actual'].values
y_ma = df['MA'].values
y_es = df['ES'].values
y_lr = df['LR'].values
y_eq = df['Equal_Average'].values
y_gwo = df['GWO_Ensemble'].values

# Calculate Absolute Percentage Errors (APE) for each sample
ape_ma = np.abs((y_true - y_ma) / y_true)
ape_es = np.abs((y_true - y_es) / y_true)
ape_lr = np.abs((y_true - y_lr) / y_true)
ape_eq = np.abs((y_true - y_eq) / y_true)
ape_gwo = np.abs((y_true - y_gwo) / y_true)

# Calculate Absolute Errors (AE)
ae_ma = np.abs(y_true - y_ma)
ae_gwo = np.abs(y_true - y_gwo)
ae_eq = np.abs(y_true - y_eq)

# Calculate Squared Errors (SE)
se_ma = (y_true - y_ma)**2
se_gwo = (y_true - y_gwo)**2
se_eq = (y_true - y_eq)**2

print("=== STATISTICAL SIGNIFICANCE TESTS (335 TEST SAMPLES) ===\n")

# 1. Paired t-test on APE (Best Baseline MA vs GWO Ensemble)
t_stat_ape, p_val_t_ape = stats.ttest_rel(ape_ma, ape_gwo)
print("1. Paired t-test on Absolute Percentage Errors (MA vs GWO Ensemble):")
print(f"   t-statistic = {t_stat_ape:.4f}, p-value = {p_val_t_ape:.4e}")

# Paired t-test on Equal Average vs GWO Ensemble
t_stat_eq, p_val_t_eq = stats.ttest_rel(ape_eq, ape_gwo)
print(f"   (Equal Average vs GWO Ensemble) t-stat = {t_stat_eq:.4f}, p-value = {p_val_t_eq:.4e}\n")

# 2. Wilcoxon Signed-Rank Test (Non-parametric paired test)
w_stat_ape, p_val_w_ape = stats.wilcoxon(ape_ma, ape_gwo)
print("2. Wilcoxon Signed-Rank Test (MA vs GWO Ensemble):")
print(f"   statistic = {w_stat_ape:.4f}, p-value = {p_val_w_ape:.4e}")

w_stat_eq, p_val_w_eq = stats.wilcoxon(ape_eq, ape_gwo)
print(f"   (Equal Average vs GWO Ensemble) statistic = {w_stat_eq:.4f}, p-value = {p_val_w_eq:.4e}\n")

# 3. Diebold-Mariano (DM) Test
# DM test compares loss differential d_t = loss(e_1t) - loss(e_2t)
def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion='MSE'):
    if criterion == 'MSE':
        e1 = (y_true - y_pred1)**2
        e2 = (y_true - y_pred2)**2
    elif criterion == 'MAE':
        e1 = np.abs(y_true - y_pred1)
        e2 = np.abs(y_true - y_pred2)
    elif criterion == 'MAPE':
        e1 = np.abs((y_true - y_pred1) / y_true)
        e2 = np.abs((y_true - y_pred2) / y_true)
    
    d = e1 - e2
    n = len(d)
    mean_d = np.mean(d)
    
    # Autocovariance for lag 0 to h-1 (Harvey et al., 1997 small sample correction)
    gamma_0 = np.var(d, ddof=0)
    
    # Standard error of mean differential
    # For h=1, SE = sqrt(gamma_0 / n)
    se_d = np.sqrt(gamma_0 / n)
    dm_stat = mean_d / se_d
    
    # Harvey, Leybourne and Newbold (HLN) adjusted DM statistic
    hln_stat = dm_stat * np.sqrt((n + 1 - 2*h + (h/n)*(h-1)) / n)
    p_val = 2 * (1 - stats.norm.cdf(abs(hln_stat)))
    return mean_d, hln_stat, p_val

print("3. Diebold-Mariano (DM) Test:")
for crit in ['MAPE', 'MAE', 'MSE']:
    m_d, dm_s, p_v = diebold_mariano_test(y_true, y_ma, y_gwo, criterion=crit)
    print(f"   [MA vs GWO Ensemble - {crit}] Mean Diff = {m_d:.6f}, DM-stat = {dm_s:.4f}, p-value = {p_v:.4e}")

for crit in ['MAPE', 'MAE', 'MSE']:
    m_d, dm_s, p_v = diebold_mariano_test(y_true, y_eq, y_gwo, criterion=crit)
    print(f"   [Equal Avg vs GWO Ensemble - {crit}] Mean Diff = {m_d:.6f}, DM-stat = {dm_s:.4f}, p-value = {p_v:.4e}")

# 4. 95% Confidence Intervals for MAPE
print("\n4. 95% Confidence Intervals for MAPE:")
def compute_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h, h

ma_m, ma_low, ma_high, ma_h = compute_ci(ape_ma * 100)
gwo_m, gwo_low, gwo_high, gwo_h = compute_ci(ape_gwo * 100)
diff_m, diff_low, diff_high, diff_h = compute_ci((ape_ma - ape_gwo) * 100)

print(f"   MA MAPE: {ma_m:.4f}% (95% CI: [{ma_low:.4f}%, {ma_high:.4f}%])")
print(f"   GWO MAPE: {gwo_m:.4f}% (95% CI: [{gwo_low:.4f}%, {gwo_high:.4f}%])")
print(f"   MAPE Difference (MA - GWO): {diff_m:.4f}% (95% CI: [{diff_low:.4f}%, {diff_high:.4f}%])")
