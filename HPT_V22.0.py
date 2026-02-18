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

# --- 核心參數與物理模型 ---
C_LIGHT = 299792.458
OMEGA_M_FIXED = 0.334  # 根據 Pantheon+ 官方設定
H0_NULL = 70.0         # 假設基準宇宙 H0 為 70 (無演化)

def get_dl_null(z, h0):
    """計算標準 LCDM 的光度距離 (無演化)"""
    integrand = lambda zp: 1.0 / np.sqrt(OMEGA_M_FIXED * (1+zp)**3 + (1 - OMEGA_M_FIXED))
    res, _ = quad(integrand, 0, z)
    return (1 + z) * (C_LIGHT / h0) * res

def run_hia_mock_test(file_name, n_mocks=1000):
    print(f"--- 啟動 HIA v22.0: {n_mocks} 次模擬檢驗 ---")
    
    # 1. 加載真實數據特徵
    df = pd.read_csv(file_name, sep=r'\s+', comment='#', engine='python')
    z_obs = df['zHD'].values
    # 這裡假設使用對角誤差加上我們測得的 Intrinsic Scatter (0.1446)
    mu_err_total = np.sqrt(df['MU_ERR_DIAG'].values**2 + 0.1446**2) 
    
    # 2. 生成理論上的「零假設」MU (無演化宇宙)
    print("正在生成零假設基準線...")
    mu_null = np.array([5 * np.log10(get_dl_null(zi, H0_NULL)) + 25 for zi in z_obs])

    # 3. 擬合函數 (用於測試 Mock 數據中是否會出現虛假斜率)
    def fit_slope(z, mu, err):
        def nll(params):
            h0_base, slope = params
            if not (60 < h0_base < 80 and -20 < slope < 30): return 1e15
            # 使用快速距離近似進行 1000 次迭代 (加速模擬)
            mu_model = 5 * np.log10((C_LIGHT * z / (h0_base + slope * z)) * (1 + 0.775 * z)) + 25
            return 0.5 * np.sum(((mu - mu_model) / err)**2)
        
        res = minimize(nll, [70, 0], method='L-BFGS-B')
        return res.x[1] # 返回擬合出的斜率

    # 4. 執行 1000 次蒙地卡羅
    mock_slopes = []
    observed_slope = 11.69  # 我們之前的實測值
    
    print(f"開始執行 {n_mocks} 次隨機宇宙模擬...")
    for i in range(n_mocks):
        # 在無演化的理論 MU 上加入符合觀測誤差的高斯噪聲
        mu_mock = mu_null + np.random.normal(0, mu_err_total)
        s = fit_slope(z_obs, mu_mock, mu_err_total)
        mock_slopes.append(s)
        if (i+1) % 100 == 0: print(f"已完成 {i+1} 次模擬...")

    # 5. 結果分析
    mock_slopes = np.array(mock_slopes)
    p_value = np.sum(mock_slopes >= observed_slope) / n_mocks
    mean_mock = np.mean(mock_slopes)
    std_mock = np.std(mock_slopes)
    
    print(f"\n--- 模擬結果報告 ---")
    print(f"模擬平均斜率: {mean_mock:.4f}")
    print(f"模擬斜率標準差 (Sigma_noise): {std_mock:.4f}")
    print(f"實測斜率 (11.69) 的 Z-score: {(observed_slope - mean_mock) / std_mock:.2f} sigma")
    print(f"P-value (隨機產生該斜率的機率): {p_value}")

    # 6. 繪圖
    plt.figure(figsize=(10, 6))
    plt.hist(mock_slopes, bins=30, alpha=0.7, color='gray', label='Mock Universes (Null Hypo)')
    plt.axvline(observed_slope, color='red', linestyle='--', label=f'Observed Slope (11.69)')
    plt.title(f"HIA v22.0: Monte Carlo Null Test ({n_mocks} runs)")
    plt.xlabel("H0 Slope (km/s/Mpc/z)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_hia_mock_test('Pantheon+SH0ES.dat', n_mocks=1000)

