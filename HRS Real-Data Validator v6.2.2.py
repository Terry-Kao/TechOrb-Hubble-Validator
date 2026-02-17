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
import urllib.parse

# --- 1. 強化後的數據獲取函數 (解決 URL 編碼問題) ---

def get_pantheon_plus_data():
    print("[*] 正在精確連接 Pantheon+ 官方資料庫 (v2022)...")
    
    # 這是經過轉義後的正確路徑，%2B 代表 '+'
    base_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    path = "Pantheon%2B_Data/4_SHOES/Pantheon%2B_SH0ES.dat"
    full_url = base_url + path
    
    try:
        # 使用自定義 Header 模擬瀏覽器，防止被 GitHub 阻擋
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(full_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # 讀取數據：官方格式為空格分隔，帶有標題
        # 使用 sep='\s+' 處理不規則空格
        data = pd.read_csv(io.StringIO(response.text), sep='\s+')
        
        # 關鍵數據過濾與清洗
        # 我們需要 zHD (哈伯圖紅移), m_b_corr (修正後的星等), m_b_corr_err_DIAG (誤差)
        # 過濾掉 z < 0.01 的近場干擾
        mask = (data['zHD'] > 0.01) & (data['IS_DIST_CAND'] > 0)
        z_obs = data['zHD'][mask].values
        mb_obs = data['m_b_corr'][mask].values
        mb_err = data['m_b_corr_err_DIAG'][mask].values
        
        print(f"    -> [成功] 已從官方路徑抓取 {len(z_obs)} 顆超新星真實數據。")
        return z_obs, mb_obs, mb_err
        
    except Exception as e:
        print(f"[!] 下載依然失敗: {e}")
        print("    -> 備案：嘗試自動編碼路徑...")
        # 最後的嘗試：自動編碼
        try:
            alt_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_SHOES/Pantheon%2B_SH0ES.dat"
            response = requests.get(alt_url, timeout=10)
            data = pd.read_csv(io.StringIO(response.text), sep='\s+')
            mask = (data['zHD'] > 0.01)
            return data['zHD'][mask].values, data['m_b_corr'][mask].values, data['m_b_corr_err_DIAG'][mask].values
        except:
            print("    -> [錯誤] 無法取得線上數據，請檢查網絡環境。")
            return None, None, None

# --- 2. 物理模型核心 (不變) ---

def theory_distance_modulus(z, h0, om, alpha=0, beta=0, model='lcdm'):
    c = 299792.458
    dz = 0.01
    z_max = np.max(z)
    z_integ = np.arange(0, z_max + dz, dz)
    
    # 宇宙背景演化 E(z)
    Ez_sq = om * (1 + z_integ)**3 + (1 - om)
    
    if model == 'lcdm':
        h_vals = h0 * np.sqrt(Ez_sq)
    else:
        # HRS 混合模型：H(z) = H_LCDM * [1 + beta * sech(alpha * ln(1+z))]
        chi_vals = np.log(1 + z_integ)
        correction = 1.0 + beta * (1.0 / np.cosh(alpha * chi_vals))
        h_vals = h0 * np.sqrt(Ez_sq) * correction
        
    inv_h = 1.0 / h_vals
    dc = np.cumsum(inv_h) * dz * c
    dc_interp = np.interp(z, z_integ, dc)
    dl = (1 + z) * dc_interp
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

# --- 3. 似然函數 (帶入哈伯張力約束) ---

def log_likelihood(theta, z, mu, err, model_type):
    if model_type == 'lcdm':
        h0, om = theta
        alpha, beta = 0, 0
    else:
        h0, om, alpha, beta = theta
        
    if not (60 < h0 < 85 and 0.1 < om < 0.5): return -np.inf
    if model_type == 'hrs' and not (0 < alpha < 5.0 and -0.5 < beta < 0.5): return -np.inf
    
    # 普朗克約束 (Omega_m = 0.315 ± 0.007)
    prior_om = -0.5 * ((om - 0.315) / 0.007)**2
    
    mu_model = theory_distance_modulus(z, h0, om, alpha, beta, model_type)
    
    # 邊際化絕對星等偏移 (Marginalizing M)
    diff = mu - mu_model
    offset = np.mean(diff)
    chisq = np.sum(((diff - offset) / err)**2)
    
    return -0.5 * chisq + prior_om

# --- 4. 執行與分析 ---

def run_final_check():
    z_obs, mb_obs, mb_err = get_pantheon_plus_data()
    if z_obs is None: return

    # 抽取 500 個點進行壓力測試
    idx = np.random.choice(len(z_obs), 500, replace=False)
    z, mu, err = z_obs[idx], mb_obs[idx], mb_err[idx]

    nwalkers, steps = 32, 600
    print("\n[*] 進入 MCMC 壓力測試階段...")

    # LCDM 測試
    sampler_l = emcee.EnsembleSampler(nwalkers, 2, log_likelihood, args=(z, mu, err, 'lcdm'))
    sampler_l.run_mcmc([73.0, 0.31] + 1e-3*np.random.randn(nwalkers, 2), steps, progress=True)

    # HRS 測試
    sampler_h = emcee.EnsembleSampler(nwalkers, 4, log_likelihood, args=(z, mu, err, 'hrs'))
    sampler_h.run_mcmc([73.0, 0.31, 1.5, 0.05] + 1e-3*np.random.randn(nwalkers, 4), steps, progress=True)

    def get_stats(sampler, k):
        lp = sampler.get_log_prob(discard=100, flat=True)
        best_lp = np.max(lp)
        aic = 2*k - 2*best_lp
        return aic, sampler.get_chain(discard=100, flat=True)[np.argmax(lp)]

    aic_l, theta_l = get_stats(sampler_l, 2)
    aic_h, theta_h = get_stats(sampler_h, 4)
    
    print("\n" + "="*50)
    print("      HRS v6.2.2 最終決戰報告 (Pantheon+ Real)")
    print("="*50)
    print(f" Delta AIC: {aic_l - aic_h:.2f}")
    print(f" [解釋] 正值表示全息修正比傳統模型更能解釋觀測張力。")
    print(f" HRS H0   : {theta_h[0]:.3f}")
    print(f" HRS Beta : {theta_h[3]:.4f}")
    print("="*50)

    # 繪圖
    flat_samples = sampler_h.get_chain(discard=100, flat=True)
    fig = corner.corner(flat_samples, labels=["$H_0$", "$\Omega_m$", "$\\alpha$", "$\\beta$"], truths=theta_h)
    plt.savefig("hrs_v6_2_2_final.png")
    print("[🎉] 最終 Corner Plot 已儲存。")

if __name__ == "__main__":
    run_final_check()
    
