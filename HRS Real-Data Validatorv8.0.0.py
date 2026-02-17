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
from scipy.integrate import quad
import os

# ==========================================
# 1. 核心動力學：上帝科技球標度變換矩陣
# ==========================================
def E_inv_v8(z, om, xi, z_t):
    # 背景為標準 LCDM，不增加任何額外能量 rho
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 【v8.0.0 核心創新】：時空標度變換矩陣 M_orb
    # 這不是能量，而是對時間流速（H）的幾何修正
    M_orb = 1.0 + xi * np.exp(-z / z_t)
    
    H_obs = E_lcdm * M_orb
    return 1.0 / np.sqrt(max(H_obs**2, 1e-10))

def get_dl_v8(z_list, om, xi, z_t):
    c = 299792.458
    dl_list = []
    for z in z_list:
        # 這裡積分的是被標度變換後的時空路徑
        integral, _ = quad(E_inv_v8, 0, z, args=(om, xi, z_t))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 數據載入
# ==========================================
def load_data_v8():
    print("[*] v8.0.0 啟動：全息標度變換矩陣 (幾何架構驗證)")
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)): return None, None, None
    
    df = pd.read_csv(dat_file, sep=r'\s+')
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    cov_matrix = raw_data[1:].reshape((n_header, n_header))
    
    return df['zHD'].values, df['m_b_corr'].values, np.linalg.inv(cov_matrix + np.eye(len(df)) * 1e-5)

# ==========================================
# 3. 似然函數
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; xi, z_t = 0.0, 0.05
    else:
        om, xi, z_t = theta
    
    # 限制範圍：保持物質密度在觀測公認範圍，驗證幾何修正的純粹性
    if not (0.27 < om < 0.33): return -np.inf
    if model == 'hrs':
        if not (0.0 < xi < 0.2): return -np.inf # 幾何增益必須為正（解決張力）
        if not (0.01 < z_t < 0.2): return -np.inf # 轉折點不宜過深

    dl = get_dl_v8(z_obs, om, xi, z_t)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    return -0.5 * np.dot(diff-delta, np.dot(inv_cov, diff-delta))

# ==========================================
# 4. 執行
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v8()
    if z is not None:
        nwalkers, steps = 32, 600
        print(f"[*] 正在計算 {len(z)} 顆超新星在標度矩陣下的映射...")
        
        # HRS v8.0 (3 參數)
        sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        pos = [0.30, 0.08, 0.05] + 1e-3*np.random.randn(nwalkers, 3)
        sampler.run_mcmc(pos, steps, progress=True)

        flat_samples = sampler.get_chain(discard=200, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=200, flat=True))
        om_b, xi_b, zt_b = flat_samples[best_idx]
        
        max_lp = np.max(sampler.get_log_prob(discard=200, flat=True))
        bic_hrs = 3 * np.log(len(z)) - 2 * max_lp 
        
        # 基準 LCDM
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.31 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        max_lp_l = np.max(sampler_l.get_log_prob(discard=200, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l
        
        print("\n" + "="*50)
        print("   HRS v8.0.0 全息標度變換報告")
        print("="*50)
        print(f" Delta BIC (架構優越性): {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 幾何增益 xi    : {xi_b:.4f}")
        print(f" 映射深度 z_t   : {zt_b:.4f}")
        print(f" 物質密度 Om    : {om_b:.4f}")
        print("="*50)

