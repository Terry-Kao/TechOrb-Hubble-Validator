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
# 1. 核心動力學：局部全息修正方程
# ==========================================
def E_inv(z, om, oh0, delta_a):
    a = 1.0 / (1.0 + z)
    # 基準 LCDM 部分
    ol = 1.0 - om
    # 局部全息修正項 (z < 0.1 活性最強)
    a_trans = 1.0 / (1.0 + 0.1) # 鎖定在 z = 0.1 轉折
    oh_a = oh0 / (1.0 + np.exp((a_trans - a) / delta_a))
    
    # 這裡假設 oh_a 是一種額外的流體密度
    E_sq = om * (1+z)**3 + ol + oh_a
    return 1.0 / np.sqrt(max(E_sq, 1e-10))

def get_dl(z_list, om, oh0, delta_a):
    c = 299792.458
    # 積分求光度距離 dl
    dl_list = []
    for z in z_list:
        integral, _ = quad(E_inv, 0, z, args=(om, oh0, delta_a))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 數據載入 (全集測試：包含 z < 0.1)
# ==========================================
def load_data_v7():
    print("[*] v7.0.0 啟動：局部全息層 (Full Sample Test)")
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)): return None, None, None
    
    df = pd.read_csv(dat_file, sep=r'\s+')
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    cov_matrix = raw_data[1:].reshape((n_header, n_header))
    
    # 不再進行 z 篩選，使用全數據來檢驗局部修正的威力
    z_obs = df['zHD'].values
    mu_obs = df['m_b_corr'].values
    inv_cov = np.linalg.inv(cov_matrix + np.eye(len(z_obs)) * 1e-5)
    return z_obs, mu_obs, inv_cov

# ==========================================
# 3. 似然函數
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; oh0, delta_a = 0.0, 0.01
    else:
        om, oh0, delta_a = theta
    
    # Priors
    if not (0.25 < om < 0.35): return -np.inf
    if model == 'hrs':
        if not (-0.2 < oh0 < 0.2): return -np.inf # 局部能量修正不宜過大
        if not (0.001 < delta_a < 0.05): return -np.inf # 邊界必須夠陡峭

    dl = get_dl(z_obs, om, oh0, delta_a)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    return -0.5 * np.dot(diff-delta, np.dot(inv_cov, diff-delta))

# ==========================================
# 4. 執行與三方驗證
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v7()
    if z is not None:
        nwalkers, steps = 32, 600
        print(f"[*] 正在驗證 {len(z)} 顆超新星的局部全息效應...")
        
        # HRS v7.0 (3 參數: Om, Oh0, delta_a)
        sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        pos = [0.30, 0.02, 0.01] + 1e-3*np.random.randn(nwalkers, 3)
        sampler.run_mcmc(pos, steps, progress=True)

        flat_samples = sampler.get_chain(discard=200, flat=True)
        best_idx = np.argmax(sampler.get_log_prob(discard=200, flat=True))
        om_b, oh0_b, da_b = flat_samples[best_idx]
        
        max_lp = np.max(sampler.get_log_prob(discard=200, flat=True))
        bic_hrs = 3 * np.log(len(z)) - 2 * max_lp 
        
        # 基準 LCDM (1 參數: Om)
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.31 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        max_lp_l = np.max(sampler_l.get_log_prob(discard=200, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l
        
        print("\n" + "="*50)
        print("   HRS v7.0.0 局部全息層最終報告 (全集測試)")
        print("="*50)
        print(f" Delta BIC (正值代表超越 LCDM): {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 局部全息能 Ωh0  : {oh0_b:.4f}")
        print(f" 邊界平滑度 Δa   : {da_b:.4f}")
        print(f" 物質密度 Om     : {om_b:.4f}")
        print("="*50)
        print(" [分析] 若 Ωh0 > 0 且 ΔBIC > 0，則證實「上帝科技球」是哈伯張力的幾何解。")

