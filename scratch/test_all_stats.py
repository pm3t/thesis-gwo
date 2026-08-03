import pandas as pd
import numpy as np
import math

df = pd.read_csv(r"c:\GWO\Result.csv")

y_true = df['Actual'].values
y_ma = df['MA'].values
y_es = df['ES'].values
y_lr = df['LR'].values
y_eq = df['Equal_Average'].values
y_gwo = df['GWO_Ensemble'].values

# Absolute errors (e = |y_true - y_pred|)
ae_ma = np.abs(y_true - y_ma)
ae_gwo = np.abs(y_true - y_gwo)
ae_eq = np.abs(y_true - y_eq)

# Absolute percentage errors (APE = |y_true - y_pred| / y_true)
ape_ma = np.abs((y_true - y_ma) / y_true)
ape_gwo = np.abs((y_true - y_gwo) / y_true)
ape_eq = np.abs((y_true - y_eq) / y_true)

# Squared errors (SE = (y_true - y_pred)^2)
se_ma = (y_true - y_ma)**2
se_gwo = (y_true - y_gwo)**2
se_eq = (y_true - y_eq)**2

n = len(y_true)

def analyze_diff(name_comp, arr1, arr2):
    diff = arr1 - arr2 # positive means arr1 has higher error than arr2 (arr2 is better)
    m_diff = np.mean(diff)
    s_diff = np.std(diff, ddof=1)
    se_diff = s_diff / math.sqrt(n)
    t_stat = m_diff / se_diff
    
    # Approx p-value for large N=335 using Normal CDF approximation (since N=335 >> 30)
    # 2 * (1 - norm_cdf(|t|))
    # Approximation for erfc
    def erfc(x):
        # Handbook of Mathematical Functions formula 7.1.26
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        sign = 1
        if x < 0: sign = -1
        x = abs(x)
        t = 1.0 / (1.0 + p*x)
        y = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t*math.exp(-x*x))
        return 1.0 - sign*y

    def norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    p_val = 2.0 * (1.0 - norm_cdf(abs(t_stat)))

    ci_low = m_diff - 1.95996 * se_diff
    ci_high = m_diff + 1.95996 * se_diff

    return {
        'comp': name_comp,
        'mean_diff': m_diff,
        'std_diff': s_diff,
        'se_diff': se_diff,
        't_stat': t_stat,
        'p_val': p_val,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

print("=== STATISTICAL SIGNIFICANCE TESTS RESULTS ===")

res_ma_gwo_ape = analyze_diff("MA vs GWO Ensemble (APE)", ape_ma, ape_gwo)
res_ma_gwo_ae = analyze_diff("MA vs GWO Ensemble (AE / MAE)", ae_ma, ae_gwo)
res_ma_gwo_se = analyze_diff("MA vs GWO Ensemble (SE / MSE)", se_ma, se_gwo)

res_eq_gwo_ape = analyze_diff("Equal Avg vs GWO Ensemble (APE)", ape_eq, ape_gwo)
res_eq_gwo_ae = analyze_diff("Equal Avg vs GWO Ensemble (AE / MAE)", ae_eq, ae_gwo)

print(f"\n1. Paired t-test / Diebold-Mariano (Seasonal MA vs GWO Ensemble):")
print(f"   APE (MAPE diff): Mean Diff = {res_ma_gwo_ape['mean_diff']*100:.4f}%, t-stat = {res_ma_gwo_ape['t_stat']:.4f}, p-value = {res_ma_gwo_ape['p_val']:.4e}")
print(f"   95% CI for MAPE diff: [{res_ma_gwo_ape['ci_low']*100:.4f}%, {res_ma_gwo_ape['ci_high']*100:.4f}%]")

print(f"\n   AE (MAE diff): Mean Diff = {res_ma_gwo_ae['mean_diff']:.6f}, t-stat = {res_ma_gwo_ae['t_stat']:.4f}, p-value = {res_ma_gwo_ae['p_val']:.4e}")
print(f"   95% CI for MAE diff: [{res_ma_gwo_ae['ci_low']:.6f}, {res_ma_gwo_ae['ci_high']:.6f}]")

print(f"\n   SE (MSE diff): Mean Diff = {res_ma_gwo_se['mean_diff']:.6f}, t-stat = {res_ma_gwo_se['t_stat']:.4f}, p-value = {res_ma_gwo_se['p_val']:.4e}")
print(f"   95% CI for MSE diff: [{res_ma_gwo_se['ci_low']:.6f}, {res_ma_gwo_se['ci_high']:.6f}]")

print(f"\n2. Paired t-test / Diebold-Mariano (Equal Average Ensemble vs GWO Ensemble):")
print(f"   APE (MAPE diff): Mean Diff = {res_eq_gwo_ape['mean_diff']*100:.4f}%, t-stat = {res_eq_gwo_ape['t_stat']:.4f}, p-value = {res_eq_gwo_ape['p_val']:.4f}")
print(f"   AE (MAE diff): Mean Diff = {res_eq_gwo_ae['mean_diff']:.6f}, t-stat = {res_eq_gwo_ae['t_stat']:.4f}, p-value = {res_eq_gwo_ae['p_val']:.4f}")

# Non-parametric Wilcoxon Signed-Rank Test on absolute errors (MAE)
def wilcoxon_test(arr1, arr2):
    d = arr1 - arr2
    d_nonzero = d[d != 0]
    ranks = pd.Series(np.abs(d_nonzero)).rank(method='average')
    pos_rank = np.sum(ranks[d_nonzero > 0])
    neg_rank = np.sum(ranks[d_nonzero < 0])
    w = min(pos_rank, neg_rank)
    n_w = len(d_nonzero)
    mean_w = n_w * (n_w + 1) / 4.0
    var_w = n_w * (n_w + 1) * (2 * n_w + 1) / 24.0
    z_w = (w - mean_w) / math.sqrt(var_w)
    def norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    p_w = 2.0 * norm_cdf(z_w) # lower tail
    return w, z_w, p_w

w_stat, z_w, p_w = wilcoxon_test(ae_ma, ae_gwo)
print(f"\n3. Wilcoxon Signed-Rank Test on Absolute Errors (MA vs GWO Ensemble):")
print(f"   W-statistic = {w_stat:.1f}, Z-score = {z_w:.4f}, p-value = {p_w:.4e}")
