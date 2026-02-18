import subprocess
import sys

# --- 自動環境檢查機制 ---
def setup_environment():
    required = {"numpy", "pandas", "matplotlib", "scipy", "requests", "emcee", "corner", "torch"}
    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        if missing:
            print(f"[*] 偵測到缺失組件: {missing}，正在自動安裝...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception:
        # 針對 Colab 環境的相容處理
        pass

setup_environment()

# --- 正式導入 ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

# --- 核心參數 ---
C_LIGHT = 299792.458
OMEGA_M_FIXED = 0.334  
H0_NULL = 70.0         

def get_dl_null(z, h0):
    integrand = lambda zp: 1.0 / np.sqrt(OMEGA_M_FIXED * (1+zp)**3 + (1 - OMEGA_M_FIXED))
    res, _ = quad(integrand, 0, z)
    return (1 + z) * (C_LIGHT / h0) * res

def run_hia_mock_test(file_name, n_mocks=1000):
    print(f"--- 啟動 HIA v22.0: {n_mocks} 次模擬檢驗 (修復版) ---")
    
    # 1. 加載數據並自動偵測欄位
    df = pd.read_csv(file_name, sep=r'\s+', comment='#', engine='python')
    
    # 自動映射欄位名稱 (適應不同版本的 Pantheon+ 格式)
    col_map = {
        'z': ['zHD', 'zcmb', 'z'],
        'mu': ['MU_SH0ES', 'mu', 'MU'],
        'err': ['MU_ERR_DIAG', 'MU_ERR', 'm_b_corr_err_RAW', 'ERR']
    }
    
    found_cols = {}
    for key, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                found_cols[key] = c
                break
        if key not in found_cols:
            raise KeyError(f"無法在數據文件中找到關鍵欄位: {key} (嘗試過 {candidates})")

    z_obs = df[found_cols['z']].values
    mu_obs_real = df[found_cols['mu']].values
    # 加入我們測得的 Intrinsic Scatter 0.1446
    mu_err_total = np.sqrt(df[found_cols['err']].values**2 + 0.1446**2) 
    
    print(f"欄位偵測成功: Redshift={found_cols['z']}, MU={found_cols['mu']}, Error={found_cols['err']}")

    # 2. 生成理論上的「零假設」MU (無演化宇宙)
    print("正在生成零假設基準線...")
    mu_null = np.array([5 * np.log10(get_dl_null(zi, H0_NULL)) + 25 for zi in z_obs])

    # 3. 擬合函數 (加速版，用於 1000 次迭代)
    def fit_slope(z, mu, err):
        def nll(params):
            h0_base, slope = params
            if not (60 < h0_base < 80 and -20 < slope * 2 < 40): return 1e15
            # 使用更穩定的近似式進行大規模模擬
            mu_model = 5 * np.log10((C_LIGHT * z / (h0_base + slope * z)) * (1 + 0.5*(1- (-0.55))*z)) + 25
            return 0.5 * np.sum(((mu - mu_model) / err)**2)
        
        res = minimize(nll, [72.0, 11.0], method='L-BFGS-B') # 以我們觀測值附近作為初始值
        return res.x[1]

    # 4. 執行 1000 次蒙地卡羅
    mock_slopes = []
    observed_slope = 11.6911
    
    print(f"開始執行 {n_mocks} 次隨機宇宙模擬...")
    for i in range(n_mocks):
        mu_mock = mu_null + np.random.normal(0, mu_err_total)
        s = fit_slope(z_obs, mu_mock, mu_err_total)
        mock_slopes.append(s)
        if (i+1) % 100 == 0: print(f"已完成 {i+1} 次模擬...")

    # 5. 結果分析
    mock_slopes = np.array(mock_slopes)
    p_value = np.sum(np.abs(mock_slopes) >= observed_slope) / n_mocks
    mean_mock = np.mean(mock_slopes)
    std_mock = np.std(mock_slopes)
    
    print(f"\n--- 模擬結果報告 ---")
    print(f"模擬平均斜率: {mean_mock:.4f}")
    print(f"模擬斜率標準差 (Sigma_noise): {std_mock:.4f}")
    print(f"實測斜率 (11.69) 的 Z-score: {(observed_slope - mean_mock) / std_mock:.2f} sigma")
    print(f"P-value: {p_value}")

    # 6. 繪圖
    plt.figure(figsize=(10, 6))
    plt.hist(mock_slopes, bins=35, alpha=0.7, color='steelblue', edgecolor='white', label='Mock Universes (Null)')
    plt.axvline(observed_slope, color='crimson', lw=2, linestyle='--', label=f'HIA Observed (11.69)')
    plt.title(f"HIA v22.0: Statistical Robustness Test (N={n_mocks})", fontsize=14)
    plt.xlabel("H0 Slope (km/s/Mpc/z)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 請確保數據路徑正確
    run_hia_mock_test('Pantheon+SH0ES.dat', n_mocks=1000)

