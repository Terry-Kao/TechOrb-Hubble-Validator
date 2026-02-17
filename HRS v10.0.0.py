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
# 1. 物理模型：全息熵力延遲 (Entropic Lag)
# ==========================================
def E_inv_v10(z, om, gamma, lam):
    # 背景時空：標準 LCDM
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 【v10.0.0 核心創新】：觀測者延遲修正
    # 這裡 gamma 代表局部資訊粘滯性
    # 邏輯：Local Lag 導致我們觀測到的 H 比真實的快
    lag = gamma * np.exp(-lam * z)
    
    # 確保分母不為零且物理合法
    h_correction = 1.0 / (1.0 - lag) if lag < 0.9 else 10.0
    
    H_obs = E_lcdm * h_correction
    return 1.0 / np.maximum(H_obs, 1e-10)

def get_dl_v10(z_list, om, gamma, lam):
    c = 299792.458
    dl_list = []
    for z in z_list:
        integral, _ = quad(E_inv_v10, 0, z, args=(om, gamma, lam))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 數據處理與似然函數
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; gamma, lam = 0.0, 1.0
    else:
        om, gamma, lam = theta
    
    # 先驗分佈 (Priors)
    if not (0.25 < om < 0.35): return -np.inf
    if model == 'hrs':
        if not (0.0 < gamma < 0.2): return -np.inf # 延遲係數必須為正
        if not (0.1 < lam < 20.0): return -np.inf # 衰減尺度
        
    dl = get_dl_v10(z_obs, om, gamma, lam)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    chi2 = np.dot(diff - delta, np.dot(inv_cov, diff - delta))
    return -0.5 * chi2

# ==========================================
# 3. 執行 MCMC 採樣
# ==========================================
def run_v10_analysis():
    print("[*] HRS v10.0.0: Holographic Entropic Lag (HELP) 啟動...")
    # 載入數據 (假設文件已存在)
    dat_file, cov_file = "Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"
    if not os.path.exists(dat_file): return print("數據文件缺失。")

    df = pd.read_csv(dat_file, sep=r'\s+')
    z, mu = df['zHD'].values, df['m_b_corr'].values
    raw_cov = np.fromfile(cov_file, sep=' ')[1:].reshape((len(z), len(z)))
    inv_cov = np.linalg.inv(raw_cov + np.eye(len(z)) * 1e-5)

    nwalkers, steps = 32, 1000
    
    # HRS v10.0 採樣
    sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
    p0 = [0.30, 0.05, 5.0] + 1e-4 * np.random.randn(nwalkers, 3)
    sampler.run_mcmc(p0, steps, progress=True)
    
    # 結果統計
    flat_samples = sampler.get_chain(discard=400, flat=True)
    max_lp = np.max(sampler.get_log_prob(discard=400, flat=True))
    bic_hrs = 3 * np.log(len(z)) - 2 * max_lp

    # LCDM 基準
    sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
    sampler_l.run_mcmc(0.31 + 1e-4 * np.random.randn(nwalkers, 1), 600, progress=True)
    max_lp_l = np.max(sampler_l.get_log_prob(discard=200, flat=True))
    bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_l

    print("\n" + "="*50)
    print("   HRS v10.0.0 最終診斷報告")
    print("="*50)
    print(f" Delta BIC (學術認可度): {bic_lcdm - bic_hrs:.4f}")
    print("-" * 50)
    best_fit = flat_samples[np.argmax(sampler.get_log_prob(discard=400, flat=True))]
    print(f" 熵力延遲強度 (Gamma): {best_fit[1]:.4f}")
    print(f" 資訊空間半徑 (Lambda): {best_fit[2]:.4f}")
    print(f" 物質密度背景 (Om)   : {best_fit[0]:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_v10_analysis()

