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
from scipy.integrate import odeint
import os

# ==========================================
# 1. 動力學核心：非線性熵增耦合 (Q = Gamma * H * rho_m * (1/H)^n)
# ==========================================
def cosmo_engine_v68(y, a, Gamma, n, Om0):
    rho_m, rho_hol = y
    H_sq = max(rho_m + rho_hol, 1e-10)
    H = np.sqrt(H_sq)
    
    # 【v6.8.0 核心創新】：非線性曲率項 (1/H)^n
    # 當宇宙膨脹，H 減小，這項會增大，代表全息邊界壓力在晚期加速釋放
    curvature_effect = (1.0 / H)**n
    Q = Gamma * H * rho_m * curvature_effect
    
    # 演化方程
    d_rho_m = (-3 * H * rho_m + Q) / (a * H)
    d_rho_hol = (-Q) / (a * H)
    
    return [d_rho_m, d_rho_hol]

def get_H_history_v68(Om0, Gamma, n, z_max):
    # 以 a=1 為基準點進行逆向積分 (a: 1.0 -> 0.1)
    a_steps = np.linspace(1.0, 1.0/(1.0 + z_max), 800)
    y0 = [Om0, 1.0 - Om0]
    sol = odeint(cosmo_engine_v68, y0, a_steps, args=(Gamma, n, Om0))
    H_history = np.sqrt(np.maximum(sol[:, 0] + sol[:, 1], 1e-10))
    return a_steps, H_history

# ==========================================
# 2. 數據與似然函數 (堅持 z > 0.1)
# ==========================================
def load_data_v68(z_min=0.1):
    print(f"[*] v6.8.0 啟動：非線性熵增曲率模型 (z > {z_min})")
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)): return None, None, None
    df = pd.read_csv(dat_file, sep=r'\s+')
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    cov_matrix = raw_data[1:].reshape((n_header, n_header))
    mask = (df['zHD'] > z_min)
    z_obs, mu_obs, indices = df[mask]['zHD'].values, df[mask]['m_b_corr'].values, df.index[mask].values
    cov_cut = cov_matrix[np.ix_(indices, indices)] + np.eye(len(indices)) * 1e-5
    return z_obs, mu_obs, np.linalg.inv(cov_cut)

def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; gamma, n = 0.0, 0.0
    else:
        om, gamma, n = theta
    
    if not (0.25 < om < 0.35): return -np.inf # 鎖定 v6.7.0 的籌碼區域
    if model == 'hrs' and not (-0.5 < gamma < 0.5 and -2.0 < n < 2.0): return -np.inf

    a_hist, H_hist = get_H_history_v68(om, gamma, n, np.max(z_obs)*1.2)
    z_hist = 1.0/a_hist - 1.0
    c = 299792.458
    dc_cum = np.cumsum((1.0/H_hist) * np.gradient(z_hist)) * c
    dl = (1 + z_obs) * np.interp(z_obs, z_hist, dc_cum)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    return -0.5 * np.dot(diff-delta, np.dot(inv_cov, diff-delta))

# ==========================================
# 3. 執行
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v68(z_min=0.1)
    if z is not None:
        nwalkers, steps = 32, 1000
        print("[*] 正在挑戰非線性動力學極限...")
        sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        pos = [0.30, -0.1, 0.5] + 1e-3*np.random.randn(nwalkers, 3)
        sampler.run_mcmc(pos, steps, progress=True)

        # 統計分析
        flat_samples = sampler.get_chain(discard=400, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=400, flat=True))
        om_b, gamma_b, n_b = flat_samples[best_idx]
        
        max_lp = np.max(sampler.get_log_prob(discard=400, flat=True))
        bic_hrs = 3 * np.log(len(z)) - 2 * max_lp 
        
        # 基準 LCDM
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.31 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        max_lp_l = np.max(sampler_l.get_log_prob(discard=400, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l
        
        print("\n" + "="*50)
        print("   HRS v6.8.0 熵增曲率動力學報告 (z > 0.1)")
        print("="*50)
        print(f" Delta BIC (深空): {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 交互強度 Gamma : {gamma_b:.4f}")
        print(f" 非線性指數 n    : {n_b:.4f}")
        print(f" 物質密度 Om    : {om_b:.4f}")
        print("="*50)
        print(f" [分析] 若 n > 0，證實全息交換隨宇宙膨脹而加速釋放。")

