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
import urllib.request
import io
import ssl
import sys

# --- 1. 使用強化的 urllib 引擎獲取真實數據 ---

def get_real_data_robust():
    print("[*] 正在透過 SSL 隧道存取 Pantheon+ 真實數據庫...")
    
    # 官方數據的原始位址
    url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_SHOES/Pantheon+_SH0ES.dat"
    
    # 忽略 SSL 憑證檢查（解決某些雲端環境的連線問題）
    context = ssl._create_unverified_context()
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=context, timeout=20) as response:
            content = response.read().decode('utf-8')
            data = pd.read_csv(io.StringIO(content), sep=r'\s+')
            print(f"    -> [成功] 已安全獲取 {len(data)} 條觀測紀錄。")
            
            # 清洗數據：排除 IS_DIST_CAND != 1 與 z < 0.01
            mask = (data['zHD'] > 0.01) & (data['IS_DIST_CAND'] > 0)
            return data['zHD'][mask].values, data['m_b_corr'][mask].values, data['m_b_corr_err_DIAG'][mask].values
            
    except Exception as e:
        print(f"\n[!] 自動下載失敗: {e}")
        print("-" * 50)
        print("【手動操作指示】")
        print("1. 請手動瀏覽: https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon+_Data/4_SHOES/Pantheon+_SH0ES.dat")
        print("2. 點擊 'Download Raw File' 並存為 'pantheon.dat'")
        print("3. 將檔案拖入此執行環境的左側資料夾。")
        print("-" * 50)
        
        # 嘗試讀取本地檔案
        try:
            data = pd.read_csv('pantheon.dat', sep=r'\s+')
            mask = (data['zHD'] > 0.01) & (data['IS_DIST_CAND'] > 0)
            print("[✅] 已成功讀取本地上傳的真實數據。")
            return data['zHD'][mask].values, data['m_b_corr'][mask].values, data['m_b_corr_err_DIAG'][mask].values
        except:
            print("[❌] 本地檔案不存在，中止執行以維持科學真實性。")
            sys.exit()

# --- 2. 物理模型與計算 (保持 v6.2.4 的嚴謹度) ---

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
        # HRS 全息修正公式
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
    
    # 注入普朗克觀測壓力 (Omega_m = 0.315)
    prior_om = -0.5 * ((om - 0.315) / 0.007)**2
    
    mu_model = theory_distance_modulus(z, h0, om, alpha, beta, model_type)
    diff = mu - mu_model
    offset = np.mean(diff) 
    chisq = np.sum(((diff - offset) / err)**2)
    return -0.5 * chisq + prior_om

# --- 3. 執行主程序 ---

if __name__ == "__main__":
    z_real, mu_real, err_real = get_real_data_robust()
    
    # 隨機抽取真實樣本
    np.random.seed(42)
    idx = np.random.choice(len(z_real), 500, replace=False)
    z, mu, err = z_real[idx], mu_real[idx], err_real[idx]

    print(f"[*] 正在對 {len(z)} 筆真實數據進行「哈伯張力」對抗測試...")
    
    nwalkers, steps = 32, 600
    # LCDM
    sampler_l = emcee.EnsembleSampler(nwalkers, 2, log_likelihood, args=(z, mu, err, 'lcdm'))
    sampler_l.run_mcmc([73.0, 0.31] + 1e-3*np.random.randn(nwalkers, 2), steps, progress=True)

    # HRS
    sampler_h = emcee.EnsembleSampler(nwalkers, 4, log_likelihood, args=(z, mu, err, 'hrs'))
    sampler_h.run_mcmc([73.0, 0.31, 1.5, 0.05] + 1e-3*np.random.randn(nwalkers, 4), steps, progress=True)

    def get_stats(sampler, k):
        lp = sampler.get_log_prob(discard=100, flat=True)
        best_idx = np.argmax(lp)
        return 2*k - 2*lp[best_idx], sampler.get_chain(discard=100, flat=True)[best_idx]

    aic_l, theta_l = get_stats(sampler_l, 2)
    aic_h, theta_h = get_stats(sampler_h, 4)

    print("\n" + "="*50)
    print(f"      HRS v6.2.5 決戰報告 (真實數據)")
    print("="*50)
    print(f" Delta AIC : {aic_l - aic_h:.4f}")
    print(f" HRS H0    : {theta_h[0]:.3f}")
    print(f" HRS Beta  : {theta_h[3]:.4f}")
    print(f" 最終結論  : {'[勝] 發現全息效應特徵' if aic_l - aic_h > 2 else '[平] 數據偏向傳統模型'}")
    print("="*50)

    # 繪圖
    labels = [r"$H_0$", r"$\Omega_m$", r"$\alpha$", r"$\beta$"]
    fig = corner.corner(sampler_h.get_chain(discard=100, flat=True), labels=labels, truths=theta_h)
    plt.savefig("hrs_v6_2_5_real_final.png")
    print("[🎉] 驗證圖表已儲存。")


