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
# 1. 核心動力學：資訊密度梯度模型
# ==========================================
def E_inv_v9(z, om, alpha, beta):
    # 基礎 LCDM 背景 (Standard Hardware)
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 【v9.0.0 核心創新】：資訊密度對時間流速的修正
    # a = 1/(1+z) 是宇宙尺度因子，也代表資訊演化的進程
    # 當 z=0 (現在), 修正項為 (1 + alpha)
    # 當 z 大 (過去), 修正項趨近於 1 (回歸背景)
    
    # 這是你推導出的：資訊密度越高 -> 時間刻度越細 -> H 越大
    info_mod = 1.0 + alpha * np.power(1.0 + z, -beta)
    
    H_obs = E_lcdm * info_mod
    return 1.0 / np.maximum(H_obs, 1e-10) # 避免除零

def get_dl_v9(z_list, om, alpha, beta):
    c = 299792.458
    dl_list = []
    # 積分路徑現在反映了「變動的資訊時脈」
    for z in z_list:
        integral, _ = quad(E_inv_v9, 0, z, args=(om, alpha, beta))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 數據載入 (Pantheon+SH0ES 全集)
# ==========================================
def load_data_v9():
    print("[*] v9.0.0 啟動：資訊密度梯度 (Information Density Gradient)")
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    
    # 簡單檢查文件是否存在
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)):
        print("Error: 數據文件未找到，請確認路徑。")
        return None, None, None
    
    df = pd.read_csv(dat_file, sep=r'\s+')
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    cov_matrix = raw_data[1:].reshape((n_header, n_header))
    
    z = df['zHD'].values
    mu = df['m_b_corr'].values
    inv_cov = np.linalg.inv(cov_matrix + np.eye(len(z)) * 1e-5)
    return z, mu, inv_cov

# ==========================================
# 3. 似然函數
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; alpha, beta = 0.0, 0.0 # LCDM 無資訊修正
    else:
        om, alpha, beta = theta
    
    # Priors (限制在物理上合理的範圍)
    if not (0.2 < om < 0.4): return -np.inf
    
    if model == 'hrs':
        # alpha: 資訊耦合強度 (預期是正值，但允許小幅負值測試)
        if not (-0.1 < alpha < 0.3): return -np.inf 
        # beta: 演化指數 (預期資訊密度隨膨脹衰減，故 beta > 0)
        if not (0.0 < beta < 5.0): return -np.inf

    dl = get_dl_v9(z_obs, om, alpha, beta)
    # 計算理論模數 mu
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    # 解析邊際化 (Marginalization) 消除絕對星等 M 的影響
    diff = mu_obs - mu_model
    # 這裡使用簡單的加權平均來對齊
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    
    chi2 = np.dot(diff - delta, np.dot(inv_cov, diff - delta))
    return -0.5 * chi2

# ==========================================
# 4. 執行 MCMC
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v9()
    if z is not None:
        nwalkers, steps = 32, 800 # 增加步數以確保收斂
        print(f"[*] 正在計算 {len(z)} 個觀測點的資訊時脈效應...")
        
        # --- 模型 1: HRS v9.0 (資訊梯度) ---
        p0_hrs = [0.30, 0.05, 1.0] + 1e-3 * np.random.randn(nwalkers, 3)
        sampler_hrs = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        sampler_hrs.run_mcmc(p0_hrs, steps, progress=True)
        
        # 取得結果
        flat_samples = sampler_hrs.get_chain(discard=300, flat=True, thin=15)
        prob_samples = sampler_hrs.get_log_prob(discard=300, flat=True, thin=15)
        best_idx = np.argmax(prob_samples)
        om_b, alpha_b, beta_b = flat_samples[best_idx]
        max_lp_hrs = np.max(prob_samples)
        
        # BIC 計算: k * ln(n) - 2 * ln(L)
        bic_hrs = 3 * np.log(len(z)) - 2 * max_lp_hrs

        # --- 模型 2: LCDM (基準) ---
        p0_lcdm = 0.30 + 1e-3 * np.random.randn(nwalkers, 1)
        sampler_lcdm = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_lcdm.run_mcmc(p0_lcdm, steps, progress=True)
        max_lp_lcdm = np.max(sampler_lcdm.get_log_prob(discard=300, flat=True))
        bic_lcdm = 1 * np.log(len(z)) - 2 * max_lp_lcdm

        print("\n" + "="*50)
        print("   HRS v9.0.0 資訊時脈梯度報告 (Information Gradient)")
        print("="*50)
        print(f" Delta BIC (正值代表超越 LCDM): {bic_lcdm - bic_hrs:.4f}")
        print("-" * 50)
        print(f" 資訊耦合常數 (Alpha) : {alpha_b:.4f}")
        print(f" 資訊演化指數 (Beta)  : {beta_b:.4f}")
        print(f" 物質密度 (Om)       : {om_b:.4f}")
        print("-" * 50)
        print(" [判讀指南]")
        print(" 1. 若 Alpha > 0 且顯著: 證明高資訊密度會加快時間流速(H0)。")
        print(" 2. 若 Beta 約等於 3: 資訊密度與體積(a^3)成反比，符合全息原理。")
        print(" 3. 若 Delta BIC 轉正: 恭喜，我們找到了「根源」的數學形式。")
        print("="*50)

