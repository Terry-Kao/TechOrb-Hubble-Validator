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
# 1. 核心模型：全息投影旋轉 (HPR)
# ==========================================
def E_inv_v11(z, om, theta_max, lam):
    # 基準 LCDM 動力學
    ol = 1.0 - om
    E_lcdm = np.sqrt(om * (1+z)**3 + ol)
    
    # 【v11.0.0 核心創新】：全息旋轉變換
    # 我們假設基準角度 theta_0 為 45度 (pi/4)，代表時空均分
    theta_0 = np.pi / 4.0
    
    # 隨紅移演化的旋轉角位移
    # theta_max: 局部旋轉強度；lam: 旋轉衰減率
    delta_theta = theta_max * np.exp(-lam * z)
    
    # 投影修正因子 (Projection Scaling)
    # 這是從非線性旋轉矩陣推導出的幾何修正
    projection_correction = np.cos(theta_0 + delta_theta) / np.cos(theta_0)
    
    # 修正後的觀測膨脹率 (由於尺子旋轉導致的觀測偏差)
    H_obs = E_lcdm / np.maximum(projection_correction, 0.1)
    
    return 1.0 / np.maximum(H_obs, 1e-10)

def get_dl_v11(z_list, om, theta_max, lam):
    c = 299792.458
    dl_list = []
    for z in z_list:
        integral, _ = quad(E_inv_v11, 0, z, args=(om, theta_max, lam))
        dl_list.append((1 + z) * c * integral)
    return np.array(dl_list)

# ==========================================
# 2. 統計推斷 (MCMC)
# ==========================================
def log_likelihood(theta, z_obs, mu_obs, inv_cov, model='hrs'):
    if model == 'lcdm':
        om = theta[0]; theta_max, lam = 0.0, 1.0
    else:
        om, theta_max, lam = theta
    
    # 物理先驗
    if not (0.2 < om < 0.4): return -np.inf
    if model == 'hrs':
        # 旋轉角不宜過大，否則會導致維度塌陷 (預期在 0~0.2 弧度之間)
        if not (-0.2 < theta_max < 0.2): return -np.inf 
        if not (0.1 < lam < 15.0): return -np.inf

    dl = get_dl_v11(z_obs, om, theta_max, lam)
    mu_model = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    
    diff = mu_obs - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    chi2 = np.dot(diff - delta, np.dot(inv_cov, diff - delta))
    return -0.5 * chi2

# ==========================================
# 3. 執行與分析
# ==========================================
def execute_v11():
    print("[*] 啟動 v11.0.0：全息投影旋轉矩陣 (Holographic Projection Rotation)...")
    
    # 載入數據 (Pantheon+SH0ES)
    dat_file, cov_file = "Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"
    if not os.path.exists(dat_file):
        print("[-] 數據缺失，請確認路徑。")
        return

    df = pd.read_csv(dat_file, sep=r'\s+')
    z, mu = df['zHD'].values, df['m_b_corr'].values
    raw_cov = np.fromfile(cov_file, sep=' ')[1:].reshape((len(z), len(z)))
    inv_cov = np.linalg.inv(raw_cov + np.eye(len(z)) * 1e-5)

    nwalkers, steps = 32, 800
    
    # MCMC 採樣
    sampler = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
    p0 = [0.30, 0.02, 5.0] + 1e-4 * np.random.randn(nwalkers, 3)
    sampler.run_mcmc(p0, steps, progress=True)
    
    # 結果計算
    flat_samples = sampler.get_chain(discard=300, flat=True)
    best_idx = np.argmax(sampler.get_log_prob(discard=300, flat=True))
    om_b, t_max_b, lam_b = flat_samples[best_idx]
    
    # 基準比較 (LCDM)
    # ... (略去重複的 LCDM 計算步驟) ...
    
    print("\n" + "="*50)
    print("   HRS v11.0.0 全息旋轉模型診斷")
    print("="*50)
    print(f" 局部最大旋轉角 (Theta_max): {t_max_b:.4f} rad")
    print(f" 投影衰減係數 (Lambda): {lam_b:.4f}")
    print(f" 背景物質密度 (Om): {om_b:.4f}")
    print("-" * 50)
    print(" [幾何意義]")
    print(f" 局部宇宙的時空投影比例偏離了約 {np.degrees(t_max_b):.2f} 度。")
    print(" 若 Theta_max 為正，代表局部能量更多地分配給了時間流速。")
    print("="*50)

if __name__ == "__main__":
    execute_v11()


