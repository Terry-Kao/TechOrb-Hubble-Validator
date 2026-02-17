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
# 1. 動力學引擎：能量交換 ODE
# ==========================================
def cosmo_engine(y, a, Gamma, Om0):
    # y[0] = rho_m, y[1] = rho_hol
    rho_m, rho_hol = y
    
    # 總能量密度 (假設輻射在低紅移可忽略)
    H_sq = (rho_m + rho_hol) 
    H = np.sqrt(max(H_sq, 1e-10))
    
    # 交互作用項 Q = Gamma * H * rho_hol
    Q = Gamma * H * rho_hol
    
    # 演化方程 d(rho)/da = (d(rho)/dt) / (aH)
    d_rho_m = (-3 * H * rho_m + Q) / (a * H)
    d_rho_hol = (-Q) / (a * H) # 假設全息項本身是 Lambda-like (w=-1)
    
    return [d_rho_m, d_rho_hol]

def get_H_history(Om0, Gamma, z_max):
    a_steps = np.linspace(1.0, 1.0/(1.0 + z_max), 500)
    # 初始條件 (現在 a=1): rho_m = Om0, rho_hol = (1 - Om0)
    y0 = [Om0, 1.0 - Om0]
    
    sol = odeint(cosmo_engine, y0, a_steps, args=(Gamma, Om0))
    rho_m_h, rho_hol_h = sol[:, 0], sol[:, 1]
    H_history = np.sqrt(rho_m_h + rho_hol_h)
    return a_steps, H_history

# ==========================================
# 2. 數據載入 (維持 z > 0.1 斷點測試)
# ==========================================
def load_data_v66(z_min=0.1):
    print(f"[*] v6.6.0 啟動：全息能量交換動力學測試 (z > {z_min})")
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
    if model == 'hrs' and not (-0.5 < gamma < 0.5): return -np.inf

    # 解 ODE 得到膨脹歷史
    a_hist, H_hist = get_H_history(om, gamma, np.max(z_obs)*1.1)
    z_hist = 1.0/a_hist - 1.0
    
    # 積分求光度距離 dl
    # dc = c * integral(1/H dz)
    c = 299792.458
    inv_H = 1.0 / H_hist
    # 注意 a_hist 是從 1 到 0 (降序)，所以 z_hist 是從 0 到 z_max (升序)
    # 我們需要對 z 進行梯形積分
    dc_cum = np.cumsum(inv_H * np.gradient(z_hist)) * c
    
    # 插值得到觀測點的距離
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
    z, mu, inv_cov = load_data_v66(z_min=0.1)
    if z is not None:
        nwalkers, steps = 32, 600
        
        # 這裡改用 2 參數模型：Om, Gamma
        print("[*] 正在透過 ODE 解構全息能量流...")
        sampler = emcee.EnsembleSampler(nwalkers, 2, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        pos = [0.3, 0.01] + 1e-3*np.random.randn(nwalkers, 2)
        sampler.run_mcmc(pos, steps, progress=True)

        # 獲取結果
        flat_samples = sampler.get_chain(discard=200, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=200, flat=True))
        om_b, gamma_b = flat_samples[best_idx]
        
        # 計算 BIC
        max_lp = np.max(sampler.get_log_prob(discard=200, flat=True))
        bic_hrs = 2 * np.log(len(z)) - 2 * max_lp # 2 參數
        
        # 基準 LCDM (1 參數: Om)
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.3 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        max_lp_l = np.max(sampler_l.get_log_prob(discard=200, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l
        
        print("\n" + "="*50)
        print("   HRS v6.6.0 全息能量交換報告 (z > 0.1)")
        print("="*50)
        print(f" Delta BIC: {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 能量交換率 Gamma : {gamma_b:.4f}")
        print(f" 物質密度 Om      : {om_b:.4f}")
        print("="*50)
        print(f" [分析] 若 Gamma > 0 且 Delta BIC > 0，則支持「能量驅動時間」立論。")

