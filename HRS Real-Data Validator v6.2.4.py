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
import corner
import requests
import io
import sys

# --- 1. 真實數據獲取引擎 (不再有模擬備援) ---

def get_real_pantheon_data():
    print("[*] 正在嘗試從多個學術鏡像站獲取真實 Pantheon+ 數據...")
    
    # 這裡使用三個不同的官方/學術鏡像地址
    urls = [
        # 1. 原始 GitHub Raw (嘗試轉義)
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_SHOES/Pantheon%2B_SH0ES.dat",
        # 2. 備用分支
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/master/Pantheon+_Data/4_SHOES/Pantheon+_SH0ES.dat",
        # 3. 簡化路徑 (如果前兩個失敗)
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_SHOES/Pantheon+_SH0ES.dat"
    ]
    
    data = None
    for url in urls:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = pd.read_csv(io.StringIO(r.text), sep=r'\s+')
                print(f"    -> [成功] 已連線至: {url[:60]}...")
                break
        except Exception as e:
            continue
            
    if data is None:
        print("\n[❌] 致命錯誤: 無法連接任何真實數據源！")
        print("    請確認網路環境是否能存取 raw.githubusercontent.com。")
        print("    為了保證科學嚴謹性，本程式已終止，拒絕使用模擬數據。")
        sys.exit() # 終止程式，不進行模擬

    # 清洗數據
    mask = (data['zHD'] > 0.01) & (data['IS_DIST_CAND'] > 0)
    return data['zHD'][mask].values, data['m_b_corr'][mask].values, data['m_b_corr_err_DIAG'][mask].values

# --- 2. 物理計算核心 ---

def theory_distance_modulus(z, h0, om, alpha=0, beta=0, model='lcdm'):
    c = 299792.458
    dz = 0.01
    z_max = np.max(z)
    z_integ = np.arange(0, z_max + dz, dz)
    Ez_sq = om * (1 + z_integ)**3 + (1 - om)
    
    if model == 'lcdm':
        h_vals = h0 * np.sqrt(Ez_sq)
    else:
        chi_vals = np.log(1 + z_integ)
        correction = 1.0 + beta * (1.0 / np.cosh(alpha * chi_vals))
        h_vals = h0 * np.sqrt(Ez_sq) * correction
        
    inv_h = 1.0 / h_vals
    dc = np.cumsum(inv_h) * dz * c
    dc_interp = np.interp(z, z_integ, dc)
    dl = (1 + z) * dc_interp
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

def log_likelihood(theta, z, mu, err, model_type):
    if model_type == 'lcdm':
        h0, om = theta
        alpha, beta = 0, 0
    else:
        h0, om, alpha, beta = theta
        
    if not (60 < h0 < 85 and 0.1 < om < 0.5): return -np.inf
    if model_type == 'hrs' and not (0 < alpha < 5.0 and -0.5 < beta < 0.5): return -np.inf
    
    # 強制注入 Planck 2018 觀測壓力
    prior_om = -0.5 * ((om - 0.315) / 0.007)**2
    
    mu_model = theory_distance_modulus(z, h0, om, alpha, beta, model_type)
    diff = mu - mu_model
    offset = np.mean(diff)
    chisq = np.sum(((diff - offset) / err)**2)
    return -0.5 * chisq + prior_om

# --- 3. 執行分析 ---

if __name__ == "__main__":
    z_real, mu_real, err_real = get_real_pantheon_data()
    
    # 為了統計真實性，隨機抽樣 500 個真實點位
    np.random.seed(42)
    idx = np.random.choice(len(z_real), 500, replace=False)
    z, mu, err = z_real[idx], mu_real[idx], err_real[idx]

    print(f"\n[*] 正在對 {len(z)} 個「真實」觀測點執行模型對抗測試...")
    
    nwalkers, steps = 32, 600
    sampler_l = emcee.EnsembleSampler(nwalkers, 2, log_likelihood, args=(z, mu, err, 'lcdm'))
    sampler_l.run_mcmc([73.0, 0.31] + 1e-3*np.random.randn(nwalkers, 2), steps, progress=True)

    sampler_h = emcee.EnsembleSampler(nwalkers, 4, log_likelihood, args=(z, mu, err, 'hrs'))
    sampler_h.run_mcmc([73.0, 0.31, 1.5, 0.05] + 1e-3*np.random.randn(nwalkers, 4), steps, progress=True)

    def get_stats(sampler, k):
        lp = sampler.get_log_prob(discard=100, flat=True)
        best_idx = np.argmax(lp)
        return 2*k - 2*lp[best_idx], sampler.get_chain(discard=100, flat=True)[best_idx]

    aic_l, theta_l = get_stats(sampler_l, 2)
    aic_h, theta_h = get_stats(sampler_h, 4)

    print("\n" + "="*50)
    print("      HRS v6.2.4 真實數據決戰結果")
    print("="*50)
    print(f" Delta AIC : {aic_l - aic_h:.4f}")
    print(f" HRS H0    : {theta_h[0]:.3f} km/s/Mpc")
    print(f" HRS Beta  : {theta_h[3]:.4f}")
    print(f" 結論      : {'[勝] HRS 成功解釋真實張力' if aic_l - aic_h > 2 else '[敗] 真實數據不支持 HRS 修正'}")
    print("="*50)

    # 繪圖
    labels = [r"$H_0$", r"$\Omega_m$", r"$\alpha$", r"$\beta$"]
    fig = corner.corner(sampler_h.get_chain(discard=100, flat=True), labels=labels, truths=theta_h)
    plt.savefig("hrs_v6_2_4_real_data.png")
    print("[🎉] 真實數據 Corner Plot 已儲存。")
    
