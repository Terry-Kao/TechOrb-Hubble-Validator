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
# 1. v12.0.0 核心：自適應反饋矩陣 (AFC)
# ==========================================
def feedback_correction(z, K, alpha):
    """
    K: 反饋強度 (係數)
    alpha: 資訊爆炸的斜率 (反映演化速度)
    """
    # 模擬資訊演化速度：晚期宇宙 (低 z) 資訊增加最快
    # 這裡使用一個反饋函數，模擬系統為了維持穩定而產生的補償
    info_change_rate = np.exp(-alpha * z)
    
    # 補償項：正比於資訊改變率
    # 邏輯：改變越快，補償越強 (扭曲越大)
    correction = K * info_change_rate
    return 1.0 + correction

def E_inv_v12(z, om, K, alpha):
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 套用自適應反饋
    adj = feedback_correction(z, K, alpha)
    H_obs = E_lcdm * adj
    
    return 1.0 / np.maximum(H_obs, 1e-10)

def get_dl_v12(z_list, om, K, alpha):
    c = 299792.458
    dl_list = []
    for z in z_list:
        integral, _ = quad(E_inv_v12, 0, z, args=(om, K, alpha))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 統計擬合與執行
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov):
    om, K, alpha = theta
    if not (0.2 < om < 0.45): return -np.inf
    if not (-0.5 < K < 0.5): return -np.inf  # 強度先驗
    if not (0.1 < alpha < 30.0): return -np.inf # 速率先驗

    dl = get_dl_v12(z_obs, om, K, alpha)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    chi2 = np.dot(diff - delta, np.dot(inv_cov, diff - delta))
    return -0.5 * chi2

def run_v12():
    print("[*] 啟動 v12.0.0：自適應反饋補償矩陣 (AFC)...")
    # 數據加載部分 (省略，同前)
    # ...
    
    nwalkers, steps = 32, 1000
    sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov))
    p0 = [0.31, 0.05, 5.0] + 1e-4 * np.random.randn(nwalkers, 3)
    sampler.run_mcmc(p0, steps, progress=True)
    
    # 診斷報告
    # ...
