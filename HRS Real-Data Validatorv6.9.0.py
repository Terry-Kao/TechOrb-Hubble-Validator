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
# 1. 動力學引擎：全息飽和機制 (Q = Gamma * H * rho_m * Tanh((1/H)^n))
# ==========================================
def cosmo_engine_v69(y, a, Gamma, n, Om0):
    rho_m, rho_hol = y
    H_sq = max(rho_m + rho_hol, 1e-10)
    H = np.sqrt(H_sq)
    
    # 【v6.9.0 核心創新】：使用 Tanh 進行飽和過度
    # 確保能量交換率不會像 v6.8.0 那樣發生數值爆炸
    saturation_term = np.tanh((1.0 / H)**n)
    Q = Gamma * H * rho_m * saturation_term
    
    d_rho_m = (-3 * H * rho_m + Q) / (a * H)
    d_rho_hol = (-Q) / (a * H)
    
    return [d_rho_m, d_rho_hol]

def get_H_history_v69(Om0, Gamma, n, z_max):
    # a 從 1.0 反向積分到 z_max 對應的標度因子
    a_steps = np.linspace(1.0, 1.0/(1.0 + z_max), 800)
    y0 = [Om0, 1.0 - Om0]
    # 使用 mxstep 參數增加求解器韌性
    sol = odeint(cosmo_engine_v69, y0, a_steps, args=(Gamma, n, Om0), mxstep=2000)
    H_history = np.sqrt(np.maximum(sol[:, 0] + sol[:, 1], 1e-10))
    return a_steps, H_history

# ==========================================
# 2. 數據與似然函數 (堅持 z > 0.1)
# ==========================================
def load_data_v69(z_min=0.1):
    print(f"[*] v6.9.0 啟動：全息飽和動力學模型 (z > {z_min})")
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
        om = theta[0]; gamma, n = 0.0, 1.0
    else:
        om, gamma, n = theta
    
    if not (0.28 < om < 0.33): return -np.inf # 鎖定 Om=0.3 的精確區間
    if model == 'hrs' and not (-1.0 < gamma < 1.0 and 0.1 < n < 5.0): return -np.inf

    try:
        a_hist, H_hist = get_H_history_v69(om, gamma, n, np.max(z_obs)*1.2)
        z_hist = 1.0/a_hist - 1.0
        c = 299792.458
        dc_cum = np.cumsum((1.0/H_hist) * np.gradient(z_hist)) * c
        dl = (1 + z_obs) * np.interp(z_obs, z_hist, dc_cum)
        mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
        
        diff = mu_obs - mu_model
        delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
        return -0.5 * np.dot(diff-delta, np.dot(inv_cov, diff-delta))
    except:
        return -np.inf

# ==========================================
# 3. 執行
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v69(z_min=0.1)
    if z is not None:
        nwalkers, steps = 32, 800
        print("[*] 正在積分全息飽和場...")
        sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        pos = [0.30, -0.1, 2.0] + 1e-3*np.random.randn(nwalkers, 3)
        sampler.run_mcmc(pos, steps, progress=True)

        flat_samples = sampler.get_chain(discard=300, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=300, flat=True))
        om_b, gamma_b, n_b = flat_samples[best_idx]
        
        max_lp = np.max(sampler.get_log_prob(discard=300, flat=True))
        bic_hrs = 3 * np.log(len(z)) - 2 * max_lp 
        
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.31 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        max_lp_l = np.max(sampler_l.get_log_prob(discard=300, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l
        
        print("\n" + "="*50)
        print("   HRS v6.9.0 全息飽和動力學報告 (z > 0.1)")
        print("="*50)
        print(f" Delta BIC (深空): {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 交互強度 Gamma : {gamma_b:.4f}")
        print(f" 飽和指數 n    : {n_b:.4f}")
        print(f" 物質密度 Om    : {om_b:.4f}")
        print("="*50)

