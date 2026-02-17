import subprocess
import sys

# --- 自動環境檢查機制 ---
def setup_environment():
    required = {"numpy", "pandas", "matplotlib", "scipy", "requests", "emcee", "corner"}
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
import emcee
import corner
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 專業級數據載入器 (修正 Header 位移問題)
# ==========================================
def load_official_pantheon_plus():
    print("[*] 正在載入 Pantheon+ 官方數據庫...")
    
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)):
        print("[❌] 錯誤：找不到檔案！請確保檔案已上傳且檔名完全正確。")
        return None, None, None

    # 1. 讀取觀測數據
    df = pd.read_csv(dat_file, sep=r'\s+')
    
    # 2. 讀取協方差矩陣 (處理 Header)
    print("    -> 正在讀取協方差矩陣並排除標頭...")
    # 使用 np.fromfile 並手動處理第一個元素
    raw_data = np.fromfile(cov_file, sep=' ')
    
    # 檔案的第一個數字通常是矩陣大小 (1701)
    n_header = int(raw_data[0])
    # 真正的數據是從索引 1 開始
    matrix_data = raw_data[1:]
    
    print(f"    -> 偵測到標頭 N={n_header}, 剩餘數據量={len(matrix_data)}")
    
    if len(matrix_data) != n_header * n_header:
        # 如果還是對不起來，嘗試全量讀取 (某些版本可能沒有標頭)
        if len(raw_data) == n_header * n_header:
            print("    [!] 偵測到無標頭格式，自動調整...")
            matrix_data = raw_data
        else:
            print(f"[❌] 矩陣大小不匹配：期望 {n_header**2}，實際 {len(matrix_data)}")
            return None, None, None

    cov_matrix = matrix_data.reshape((n_header, n_header))

    # 3. 數據過濾 (zHD > 0.01 是 Pantheon+ 建議的宇宙學基準)
    mask = (df['zHD'] > 0.01)
    df_clean = df[mask].reset_index(drop=True)
    z_obs = df_clean['zHD'].values
    mu_obs = df_clean['m_b_corr'].values
    
    # 同步切割矩陣
    indices = df.index[mask].values
    cov_matrix_cut = cov_matrix[np.ix_(indices, indices)]
    
    print("    -> 正在計算反矩陣 (這需要一點 CPU 效能)...")
    inv_cov = np.linalg.inv(cov_matrix_cut)
    
    print(f"[✅] 數據成功對齊：{len(z_obs)} 個點位。")
    return z_obs, mu_obs, inv_cov

# ==========================================
# 2. 物理模型與 Likelihood (穩定版)
# ==========================================
def theory_distance_modulus(z, h0, om, alpha, beta, model='lcdm'):
    c = 299792.458
    # 增加積分精度以回應 GROK 的批評
    z_integ = np.linspace(0, np.max(z)*1.05, 1000)
    Ez = np.sqrt(om * (1 + z_integ)**3 + (1 - om))
    
    if model == 'hrs':
        # 指數衰減修正：確保高紅移回歸 LCDM
        correction = 1.0 + beta * np.exp(-z_integ / alpha)
        hz = h0 * Ez * correction
    else:
        hz = h0 * Ez
        
    inv_hz = 1.0 / hz
    dc_cum = np.cumsum(inv_hz) * (z_integ[1] - z_integ[0]) * c
    dc_interp = np.interp(z, z_integ, dc_cum)
    dl = (1 + z) * dc_interp
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

def log_likelihood(theta, z, mu, inv_cov, model_type):
    if model_type == 'lcdm':
        h0, om = theta
        alpha, beta = 1.0, 0.0
    else:
        h0, om, alpha, beta = theta
    
    # 嚴格的物理先驗
    if not (65 < h0 < 80 and 0.2 < om < 0.4): return -np.inf
    if model_type == 'hrs' and not (0.01 < alpha < 3.0 and -0.2 < beta < 1.0): return -np.inf

    mu_model = theory_distance_modulus(z, h0, om, alpha, beta, model_type)
    diff = mu - mu_model
    # 矩陣卡方運算
    chisq = np.dot(diff, np.dot(inv_cov, diff))
    return -0.5 * chisq

# ==========================================
# 3. 執行分析與統計
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_official_pantheon_plus()
    
    if z is not None:
        n_data = len(z)
        nwalkers, steps = 32, 1000
        
        # --- 執行 LCDM ---
        print("\n[*] 正在執行基準模型 LCDM...")
        sampler_l = emcee.EnsembleSampler(nwalkers, 2, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc([73.0, 0.31] + 1e-3*np.random.randn(nwalkers, 2), steps, progress=True)
        
        # --- 執行 HRS ---
        print("\n[*] 正在執行全息修正模型 HRS (Evolutionary)...")
        sampler_h = emcee.EnsembleSampler(nwalkers, 4, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        sampler_h.run_mcmc([73.0, 0.31, 0.5, 0.05] + 1e-3*np.random.randn(nwalkers, 4), steps, progress=True)

        # 統計分析
        def get_metrics(sampler, k):
            lp = sampler.get_log_prob(discard=200, flat=True)
            max_log_like = np.max(lp)
            aic = 2*k - 2*max_log_like
            bic = k * np.log(n_data) - 2*max_log_like
            best_theta = sampler.get_chain(discard=200, flat=True)[np.argmax(lp)]
            return aic, bic, best_theta

        aic_l, bic_l, theta_l = get_metrics(sampler_l, 2)
        aic_h, bic_h, theta_h = get_metrics(sampler_h, 4)

        print("\n" + "="*50)
        print("   HRS v6.3.2 數據對齊驗證報告")
        print("="*50)
        print(f" Delta AIC: {aic_l - aic_h:.4f}")
        print(f" Delta BIC: {bic_l - bic_h:.4f}")
        print("-" * 50)
        print(f" HRS 最佳擬合 H0: {theta_h[0]:.3f}")
        print(f" HRS 全息強度 Beta: {theta_h[3]:.4f}")
        print("-" * 50)
        
        # 繪圖
        labels = [r"$H_0$", r"$\Omega_m$", r"$\alpha$", r"$\beta$"]
        samples = sampler_h.get_chain(discard=200, flat=True)
        fig = corner.corner(samples, labels=labels, truths=theta_h, show_titles=True)
        plt.savefig("hrs_v6_3_2_final.png")
        print("[🎉] 成功！請查看結果與圖表。")

