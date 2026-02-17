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
# 1. 核心模型：自適應反饋補償 (AFC)
# ==========================================
def stability_compensation(z, K, alpha):
    """
    K: 反饋補償強度
    alpha: 資訊爆炸的演化斜率 (代表系統對演化速率的敏感度)
    """
    # 模擬資訊產出率 (Information Production Rate)
    # 在晚期宇宙 (z 靠近 0)，演化速率達到峰值
    dI_dz = np.exp(-alpha * z)
    
    # 【v12.0.0 核心創新】：動態補償項
    # 這裡的邏輯是：系統為了補償資訊噴發帶來的擾動，
    # 會產生一個「自適應扭曲」來維持整體的穩定態。
    # 這是一個非線性的指數補償。
    return np.exp(K * dI_dz)

def E_inv_v12(z, om, K, alpha):
    # 背景時空：標準 LCDM
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 套用自適應反饋補償
    # 這裡 H_obs 是觀測者在扭曲架構下看到的膨脹率
    H_obs = E_lcdm * stability_compensation(z, K, alpha)
    
    return 1.0 / np.maximum(H_obs, 1e-10)

def get_dl_v12(z_list, om, K, alpha):
    c = 299792.458
    dl_list = []
    for z in z_list:
        integral, _ = quad(E_inv_v12, 0, z, args=(om, K, alpha))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 統計擬合函數 (Log-Likelihood)
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov):
    om, K, alpha = theta
    
    # 嚴格的物理先驗 (Priors)
    if not (0.25 < om < 0.35): return -np.inf
    # K 可以是正或負，代表「加速補償」或「延遲補償」
    if not (-0.3 < K < 0.3): return -np.inf
    # alpha 決定了補償發生的紅移範圍
    if not (0.1 < alpha < 20.0): return -np.inf

    try:
        dl = get_dl_v12(z_obs, om, K, alpha)
        mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
        
        diff = mu_obs - mu_model
        # 邊緣化哈伯常數 H0 的影響
        delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
        chi2 = np.dot(diff - delta, np.dot(inv_cov, diff - delta))
        return -0.5 * chi2
    except:
        return -np.inf

# ==========================================
# 3. 執行 MCMC 分析
# ==========================================
def run_v12_analysis():
    print("[*] HRS v12.0.0: Adaptive Feedback Compensation (AFC) 啟動...")
    
    # 數據檢查
    dat_file, cov_file = "Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"
    if not os.path.exists(dat_file):
        print("[-] 找不到 Pantheon+ 數據文件，請確認環境。")
        return

    df = pd.read_csv(dat_file, sep=r'\s+')
    z, mu = df['zHD'].values, df['m_b_corr'].values
    raw_cov = np.fromfile(cov_file, sep=' ')[1:].reshape((len(z), len(z)))
    inv_cov = np.linalg.inv(raw_cov + np.eye(len(z)) * 1e-5)

    nwalkers, steps = 32, 1000
    sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov))
    
    # 初始位置：從之前的 v11 經驗微調
    p0 = [0.30, 0.05, 10.0] + 1e-4 * np.random.randn(nwalkers, 3)
    
    print("[*] 開始馬可夫鏈蒙地卡羅 (MCMC) 採樣，請稍候...")
    sampler.run_mcmc(p0, steps, progress=True)
    
    # 獲取結果
    flat_samples = sampler.get_chain(discard=400, flat=True)
    max_lp_idx = np.argmax(sampler.get_log_prob(discard=400, flat=True))
    best_fit = flat_samples[max_lp_idx]
    
    print("\n" + "="*50)
    print("   HRS v12.0.0 自適應反饋補償報告")
    print("="*50)
    print(f" 反饋補償強度 (K)    : {best_fit[1]:.4f}")
    print(f" 資訊演化斜率 (Alpha): {best_fit[2]:.4f}")
    print(f" 物質密度背景 (Om)   : {best_fit[0]:.4f}")
    print("-" * 50)
    print(" [判讀指南]")
    print(" 1. 若 K 為正，代表系統為了應對資訊增長，產生了觀測上的加速效應。")
    print(" 2. Alpha 越高，代表這個補償越集中在現代宇宙 (低紅移)。")
    print("="*50)

if __name__ == "__main__":
    run_v12_analysis()

