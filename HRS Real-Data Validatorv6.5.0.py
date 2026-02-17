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
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 數據載入 (保持 z > 0.1 斷點測試以示嚴謹)
# ==========================================
def load_data_v65(z_min=0.1):
    print(f"[*] v6.5.0 啟動：執行「能量-時間」耦合動力學測試 (z > {z_min})")
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
# 2. 全息動力學核心 (Energy-Time Evolution)
# ==========================================
def theory_mu_dynamics(z, om, w0, wa):
    """
    om: 物質密度
    w0, wa: 全息流體的狀態方程參數 (CPL parametrization)
    這代表了能量變化如何驅動膨脹(時間流動)
    """
    c = 299792.458
    # a = 1 / (1+z)
    a_integ = np.linspace(1.0, 1.0/(1.0 + np.max(z)*1.1), 1000)
    z_integ = 1.0/a_integ - 1.0
    
    # 全息流體的能量演化項 (基於 CPL 模型演化)
    # rho_hol ~ a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
    f_a = a_integ**(-3*(1 + w0 + wa)) * np.exp(-3 * wa * (1 - a_integ))
    
    # 總哈伯演化 E(z)
    ol = 1.0 - om # 假設平坦宇宙，剩下的就是全息能量
    Ez = np.sqrt(om * (1 + z_integ)**3 + ol * f_a)
    
    # 距離積分
    inv_hz = 1.0 / Ez
    # 這裡積分需要小心，因為 z_integ 是降序
    dc_cum = np.abs(np.cumsum(inv_hz) * (z_integ[1] - z_integ[0])) * c
    dc_interp = np.interp(z, z_integ[::-1], dc_cum[::-1])
    dl = (1 + z) * dc_interp
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

# ==========================================
# 3. 似然函數 (解析邊際化)
# ==========================================
def log_likelihood(theta, z, mu, inv_cov, model_type):
    if model_type == 'lcdm':
        om = theta[0]
        w0, wa = -1.0, 0.0 # 標準暗能量
    else:
        om, w0, wa = theta
    
    # 物理先驗
    if not (0.2 < om < 0.5): return -np.inf
    if model_type == 'hrs_dyn' and not (-2.0 < w0 < -0.5 and -2.0 < wa < 2.0): return -np.inf

    mu_model = theory_mu_dynamics(z, om, w0, wa)
    diff = mu - mu_model
    delta = np.sum(np.dot(inv_cov, diff)) / np.sum(inv_cov)
    diff_corr = diff - delta
    chisq = np.dot(diff_corr, np.dot(inv_cov, diff_corr))
    return -0.5 * chisq

# ==========================================
# 4. 執行與對齊
# ==========================================
if __name__ == "__main__":
    z, mu, inv_cov = load_data_v65(z_min=0.1)
    
    if z is not None:
        nwalkers, steps = 32, 1000
        
        print("\n[*] 正在計算「全息能量-時間」演化軌跡...")
        sampler_d = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs_dyn'))
        pos = [0.3, -1.1, 0.1] + 1e-3*np.random.randn(nwalkers, 3)
        sampler_d.run_mcmc(pos, steps, progress=True)
        
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.3 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)

        # 統計分析
        def get_metrics(sampler, k):
            lp = sampler.get_log_prob(discard=300, flat=True)
            return k * np.log(len(z)) - 2*np.max(lp), sampler.get_chain(discard=300, flat=True)[np.argmax(lp)]

        bic_l, _ = get_metrics(sampler_l, 1)
        bic_d, theta_d = get_metrics(sampler_d, 3)

        print("\n" + "="*50)
        print("   HRS v6.5.0 能量-時間耦合報告 (z > 0.1)")
        print("="*50)
        print(f" Delta BIC: {bic_l - bic_d:.4f}")
        print("-" * 50)
        print(f" 最佳參數 Om:{theta_d[0]:.4f}, w0:{theta_d[1]:.4f}, wa:{theta_d[2]:.4f}")
        print(f" 結論: { '支持新動力學' if bic_l - bic_d > 0 else '仍受阻於 LCDM' }")
        print("="*50)

