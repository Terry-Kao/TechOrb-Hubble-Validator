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
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 數據過濾：中高紅移斷點測試
# ==========================================
def load_data_breakpoint(z_min=0.1): # 這裡設定斷點為 0.1
    print(f"[*] v6.3.4 斷點測試啟動：僅保留 z > {z_min} 的深空數據...")
    
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)):
        print("[❌] 錯誤：找不到數據檔案。")
        return None, None, None

    df = pd.read_csv(dat_file, sep=r'\s+')
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    cov_matrix = raw_data[1:].reshape((n_header, n_header))

    # --- 執行斷點過濾 ---
    mask = (df['zHD'] > z_min) 
    df_clean = df[mask].reset_index(drop=True)
    z_obs = df_clean['zHD'].values
    mu_obs = df_clean['m_b_corr'].values
    indices = df.index[mask].values
    cov_cut = cov_matrix[np.ix_(indices, indices)]
    
    # 正規化與求逆
    cov_cut += np.eye(len(cov_cut)) * 1e-5
    inv_cov = np.linalg.inv(cov_cut)
    
    print(f"[✅] 斷點測試準備完成：剩餘 {len(z_obs)} 個深空點位 (已剔除局部數據)。")
    return z_obs, mu_obs, inv_cov

# ==========================================
# 2. 物理模型與 Likelihood (同 v6.3.3)
# ==========================================
def theory_mu_shape(z, om, alpha, beta, model='lcdm'):
    c = 299792.458
    z_integ = np.linspace(0, np.max(z)*1.05, 1000)
    Ez = np.sqrt(om * (1 + z_integ)**3 + (1 - om))
    if model == 'hrs':
        correction = 1.0 + beta * np.exp(-z_integ / alpha)
        hz = Ez * correction
    else:
        hz = Ez
    inv_hz = 1.0 / hz
    dc_cum = np.cumsum(inv_hz) * (z_integ[1] - z_integ[0]) * c
    dc_interp = np.interp(z, z_integ, dc_cum)
    dl = (1 + z) * dc_interp
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

def log_likelihood(theta, z, mu, inv_cov, model_type):
    if model_type == 'lcdm':
        om = theta[0]; alpha = 1.0; beta = 0.0
    else:
        om, alpha, beta = theta
    if not (0.1 < om < 0.6): return -np.inf
    if model_type == 'hrs' and not (0.01 < alpha < 10.0 and -2.0 < beta < 2.0): return -np.inf

    mu_model = theory_mu_shape(z, om, alpha, beta, model_type)
    diff = mu - mu_model
    W = np.sum(inv_cov)
    delta = np.sum(np.dot(inv_cov, diff)) / W
    diff_corr = diff - delta
    chisq = np.dot(diff_corr, np.dot(inv_cov, diff_corr))
    return -0.5 * chisq

# ==========================================
# 3. 執行測試
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_breakpoint(z_min=0.1)
    
    if z is not None:
        nwalkers, steps = 32, 1000
        # LCDM
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.3 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        # HRS
        sampler_h = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        sampler_h.run_mcmc([0.3, 1.0, 0.1] + 1e-3*np.random.randn(nwalkers, 3), steps, progress=True)

        # 統計
        def get_bic(sampler, k, n_data):
            lp = sampler.get_log_prob(discard=300, flat=True)
            return k * np.log(n_data) - 2*np.max(lp), sampler.get_chain(discard=300, flat=True)[np.argmax(lp)]

        bic_l, _ = get_bic(sampler_l, 1, len(z))
        bic_h, theta_h = get_bic(sampler_h, 3, len(z))
        
        print("\n" + "="*50)
        print(f"   HRS v6.3.4 斷點測試報告 (z > 0.1)")
        print("="*50)
        print(f" Delta BIC (深空): {bic_l - bic_h:.4f}")
        print("-" * 50)
        print(f" HRS 深空最佳參數:")
        print(f" Omega_m : {theta_h[0]:.4f}")
        print(f" Alpha   : {theta_h[1]:.4f}")
        print(f" Beta    : {theta_h[2]:.4f}")
        print("="*50)

