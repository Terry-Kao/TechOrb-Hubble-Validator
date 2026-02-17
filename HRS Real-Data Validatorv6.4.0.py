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
# 1. 數據載入 (維持 v6.3.4 的深空斷點標準 z > 0.1)
# ==========================================
def load_data_wave(z_min=0.1):
    print(f"[*] v6.4.0 全息球面波啟動：斷點 z > {z_min}")
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)):
        return None, None, None

    df = pd.read_csv(dat_file, sep=r'\s+')
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    cov_matrix = raw_data[1:].reshape((n_header, n_header))

    mask = (df['zHD'] > z_min)
    z_obs = df[mask]['zHD'].values
    mu_obs = df[mask]['m_b_corr'].values
    indices = df.index[mask].values
    cov_cut = cov_matrix[np.ix_(indices, indices)] + np.eye(len(indices)) * 1e-5
    inv_cov = np.linalg.inv(cov_cut)
    
    return z_obs, mu_obs, inv_cov

# ==========================================
# 2. 全息球面波模型 (Sinc Function)
# ==========================================
def theory_mu_wave(z, om, alpha, beta, model='hrs'):
    c = 299792.458
    z_integ = np.linspace(0, np.max(z)*1.1, 1000)
    Ez = np.sqrt(om * (1 + z_integ)**3 + (1 - om))
    
    if model == 'hrs_wave':
        # 全息球面波公式: sinc(z/alpha) = sin(z/alpha) / (z/alpha)
        # 防止 z=0 除以零
        x = np.maximum(z_integ / alpha, 1e-9)
        correction = 1.0 + beta * (np.sin(x) / x)
        hz = Ez * correction
    else:
        hz = Ez
        
    inv_hz = 1.0 / hz
    dc_cum = np.cumsum(inv_hz) * (z_integ[1] - z_integ[0]) * c
    dc_interp = np.interp(z, z_integ, dc_cum)
    dl = (1 + z) * dc_interp
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

# ==========================================
# 3. 似然函數 (帶解析邊際化)
# ==========================================
def log_likelihood(theta, z, mu, inv_cov, model_type):
    if model_type == 'lcdm':
        om = theta[0]; alpha, beta = 1.0, 0.0
    else:
        om, alpha, beta = theta
    
    if not (0.2 < om < 0.5): return -np.inf
    if model_type == 'hrs_wave' and not (0.01 < alpha < 5.0 and -1.0 < beta < 1.0): return -np.inf

    mu_model = theory_mu_wave(z, om, alpha, beta, model_type)
    diff = mu - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    diff_corr = diff - delta
    chisq = np.dot(diff_corr, np.dot(inv_cov, diff_corr))
    return -0.5 * chisq

# ==========================================
# 4. 執行
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_wave(z_min=0.1)
    
    if z is not None:
        nwalkers, steps = 32, 1500 # 增加步數以確保收斂
        
        print("\n[*] 正在計算全息球面波干涉...")
        sampler_w = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs_wave'))
        # 初始值設定
        pos = [0.31, 0.5, 0.1] + 1e-3*np.random.randn(nwalkers, 3)
        sampler_w.run_mcmc(pos, steps, progress=True)
        
        # 基準模型 LCDM
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.31 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)

        # 統計
        def get_metrics(sampler, k):
            lp = sampler.get_log_prob(discard=500, flat=True)
            return k * np.log(len(z)) - 2*np.max(lp), sampler.get_chain(discard=500, flat=True)[np.argmax(lp)]

        bic_l, _ = get_metrics(sampler_l, 1)
        bic_w, theta_w = get_metrics(sampler_w, 3)

        print("\n" + "="*50)
        print("   HRS v6.4.0 全息球面波測試報告")
        print("="*50)
        print(f" Delta BIC (深空): {bic_l - bic_w:.4f}")
        print("-" * 50)
        print(f" 最佳參數 Om:{theta_w[0]:.4f}, Alpha:{theta_w[1]:.4f}, Beta:{theta_w[2]:.4f}")
        print("="*50)

