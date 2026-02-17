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
import emcee
from scipy.integrate import quad
import os

# ==========================================
# 1. 核心模型：全息諧振干涉 (HRI)
# ==========================================
def resonance_factor(z, A, B, phi):
    """
    A: 振幅 (偏差的強度)
    B: 頻率 (在 ln(1+z) 空間中的振動頻率)
    phi: 相位 (初始偏差位置)
    """
    # 使用對數紅移空間，因為宇宙演化通常是對數律的
    log_z = np.log(1.0 + z)
    # 生成干涉波
    return 1.0 + A * np.sin(B * log_z + phi)

def E_inv_v16(z, om, A, B, phi):
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 疊加諧振因子
    H_obs = E_lcdm * resonance_factor(z, A, B, phi)
    
    return 1.0 / np.maximum(H_obs, 1e-10)

def get_dl_v16(z_list, om, A, B, phi):
    c = 299792.458
    dl_list = []
    for z in z_list:
        integral, _ = quad(E_inv_v16, 0, z, args=(om, A, B, phi))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 統計推斷 (MCMC) - 找尋那根「弦」的頻率
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov):
    om, A, B, phi = theta
    
    # 物理先驗：振幅不應超過 15%，否則會破壞大尺度結構
    if not (0.25 < om < 0.35): return -np.inf
    if not (0.0 < A < 0.15): return -np.inf 
    if not (0.1 < B < 50.0): return -np.inf
    if not (0.0 < phi < 2*np.pi): return -np.inf

    try:
        dl = get_dl_v16(z_obs, om, A, B, phi)
        mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
        
        diff = mu_obs - mu_model
        delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
        chi2 = np.dot(diff - delta, np.dot(inv_cov, diff - delta))
        return -0.5 * chi2
    except:
        return -np.inf

# ==========================================
# 3. 執行分析
# ==========================================
def execute_v16():
    print("[*] 啟動 v16.0.0：全息諧振干涉掃描...")
    
    # 加載數據 (Pantheon+SH0ES)
    dat_file, cov_file = "Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"
    if not os.path.exists(dat_file): return print("[-] 數據缺失")

    df = pd.read_csv(dat_file, sep=r'\s+')
    z, mu = df['zHD'].values, df['m_b_corr'].values
    raw_cov = np.fromfile(cov_file, sep=' ')[1:].reshape((len(z), len(z)))
    inv_cov = np.linalg.inv(raw_cov + np.eye(len(z)) * 1e-5)

    nwalkers, steps = 32, 600
    sampler = emcee.EnsembleSampler(nwalkers, 4, log_likelihood, args=(z, mu, inv_cov))
    
    # 初始猜測：微小的擾動
    p0 = [0.30, 0.05, 10.0, np.pi] + 1e-4 * np.random.randn(nwalkers, 4)
    sampler.run_mcmc(p0, steps, progress=True)
    
    samples = sampler.get_chain(discard=200, flat=True)
    best_idx = np.argmax(sampler.get_log_prob(discard=200, flat=True))
    om_b, A_b, B_b, phi_b = samples[best_idx]
    
    print("\n" + "="*50)
    print("   v16.0.0 全息諧振診斷報告")
    print("="*50)
    print(f" 諧振振幅 (A): {A_b:.4f}")
    print(f" 諧振頻率 (B): {B_b:.4f}")
    print(f" 初始相位 (phi): {phi_b:.4f} rad")
    print("-" * 50)
    print(" [幾何意義]")
    print(f" 宇宙架構正在以 {B_b:.2f} 的對數頻率進行『呼吸』。")
    print(f" 我們觀測到的哈伯張力可能只是這個呼吸週期中的『吸氣』階段。")
    print("="*50)

if __name__ == "__main__":
    execute_v16()

