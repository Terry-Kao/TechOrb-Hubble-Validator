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
# 1. 動力學引擎：能量濃度掛鉤 (Q = Gamma * H * rho_m)
# ==========================================
def cosmo_engine_v67(y, a, Gamma, Om0):
    # y[0] = rho_m, y[1] = rho_hol
    rho_m, rho_hol = y
    
    # 總能量密度
    H_sq = (rho_m + rho_hol) 
    H = np.sqrt(max(H_sq, 1e-10))
    
    # 【核心修改】：能量交換 Q 與物質濃度 rho_m 成正比
    # 這代表能量變化率隨環境能量濃度而變 (時間動力隨能量演化)
    Q = Gamma * H * rho_m
    
    # 演化方程 d(rho)/da
    d_rho_m = (-3 * H * rho_m + Q) / (a * H)
    d_rho_hol = (-Q) / (a * H)
    
    return [d_rho_m, d_rho_hol]

def get_H_history_v67(Om0, Gamma, z_max):
    a_steps = np.linspace(1.0, 1.0/(1.0 + z_max), 600)
    y0 = [Om0, 1.0 - Om0]
    sol = odeint(cosmo_engine_v67, y0, a_steps, args=(Gamma, Om0))
    rho_m_h, rho_hol_h = sol[:, 0], sol[:, 1]
    H_history = np.sqrt(np.maximum(rho_m_h + rho_hol_h, 1e-10))
    return a_steps, H_history

# ==========================================
# 2. 數據載入 (堅持 z > 0.1 斷點測試)
# ==========================================
def load_data_v67(z_min=0.1):
    print(f"[*] v6.7.0 啟動：能量濃度掛鉤模型 (z > {z_min})")
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
# 3. 似然函數
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; gamma = 0.0
    else:
        om, gamma = theta
    
    if not (0.2 < om < 0.5): return -np.inf
    if model == 'hrs' and not (-1.0 < gamma < 1.0): return -np.inf

    a_hist, H_hist = get_H_history_v67(om, gamma, np.max(z_obs)*1.2)
    z_hist = 1.0/a_hist - 1.0
    
    c = 299792.458
    inv_H = 1.0 / H_hist
    dc_cum = np.cumsum(inv_H * np.gradient(z_hist)) * c
    
    dl = (1 + z_obs) * np.interp(z_obs, z_hist, dc_cum)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    chisq = np.dot(diff-delta, np.dot(inv_cov, diff-delta))
    return -0.5 * chisq

# ==========================================
# 4. 執行
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v67(z_min=0.1)
    if z is not None:
        nwalkers, steps = 32, 800
        
        print("[*] 正在計算「能量濃度-時間」耦合演化...")
        sampler = emcee.EnsembleSampler(nwalkers, 2, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        pos = [0.31, 0.05] + 1e-3*np.random.randn(nwalkers, 2)
        sampler.run_mcmc(pos, steps, progress=True)

        flat_samples = sampler.get_chain(discard=300, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=300, flat=True))
        om_b, gamma_b = flat_samples[best_idx]
        
        max_lp = np.max(sampler.get_log_prob(discard=300, flat=True))
        bic_hrs = 2 * np.log(len(z)) - 2 * max_lp 
        
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.31 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        max_lp_l = np.max(sampler_l.get_log_prob(discard=300, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l
        
        print("\n" + "="*50)
        print("   HRS v6.7.0 能量濃度交互報告 (z > 0.1)")
        print("="*50)
        print(f" Delta BIC: {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 交互耦合 Gamma : {gamma_b:.4f}")
        print(f" 物質密度 Om    : {om_b:.4f}")
        print("="*50)
        print(f" [深度分析] 若 Gamma > 0，代表全息能量隨物質密度演化而釋放。")

