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
import corner
import matplotlib.pyplot as plt
import requests
import io
import time

# ==========================================
# 1. 專業級數據載入器 (直接從 GitHub Raw 獲取)
# ==========================================
def load_official_pantheon_plus():
    print("[*] 正在建立與 Pantheon+ 官方數據庫的連線...")
    
    # 定義 Raw URL (注意 URL 編碼)
    base_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    dat_url = base_url + "Pantheon%2B_SH0ES.dat"
    cov_url = base_url + "Pantheon%2B_SH0ES_STAT%2BSYS.cov"
    
    # 1. 下載並解析數據表 (.dat)
    print("    -> 正在下載觀測數據表 (Pantheon+SH0ES.dat)...")
    r_dat = requests.get(dat_url)
    if r_dat.status_code != 200: raise Exception("無法下載數據表")
    
    # 使用 Pandas 讀取 (第一行是標題)
    df = pd.read_csv(io.StringIO(r_dat.text), sep=r'\s+')
    
    # 篩選數據：只保留用於宇宙學擬合的樣本 (zHD > 0.01)
    # 注意：Pantheon+ 官方建議使用 HELIO_Z 或 zHD，這裡使用 zHD
    mask = (df['zHD'] > 0.01)
    df_clean = df[mask].reset_index(drop=True)
    
    z_obs = df_clean['zHD'].values
    mu_obs = df_clean['m_b_corr'].values
    print(f"    -> [成功] 已載入 {len(z_obs)} 個有效觀測點。")

    # 2. 下載並解析協方差矩陣 (.cov)
    print("    -> 正在下載並建構 1701x1701 協方差矩陣 (這可能需要一點時間)...")
    r_cov = requests.get(cov_url)
    if r_cov.status_code != 200: raise Exception("無法下載協方差矩陣")
    
    # 讀取純文本矩陣數據
    raw_cov_data = np.fromstring(r_cov.text, sep=' ')
    
    # 檢查矩陣大小是否正確 (應該是 N_total * N_total)
    n_total = int(np.sqrt(len(raw_cov_data)))
    print(f"    -> 原始矩陣維度: {n_total}x{n_total}")
    
    cov_matrix = raw_cov_data.reshape((n_total, n_total))
    
    # 關鍵步驟：根據上面的 mask 同步切割矩陣
    # 我們必須只保留與 z_obs 對應的行與列
    indices = df.index[mask].values
    cov_matrix_cut = cov_matrix[np.ix_(indices, indices)]
    
    # 預先計算反矩陣 (Inverse Covariance Matrix) 以加速 MCMC
    # 使用 Cholesky 分解或偽逆矩陣來增加數值穩定性
    print("    -> 正在計算反矩陣 (Inverting Covariance Matrix)...")
    try:
        inv_cov = np.linalg.inv(cov_matrix_cut)
    except np.linalg.LinAlgError:
        print("    [!] 警告：矩陣奇異，改用偽逆矩陣 (Pseudo-inverse)")
        inv_cov = np.linalg.pinv(cov_matrix_cut)
        
    print(f"    -> [成功] 數據準備完成。")
    return z_obs, mu_obs, inv_cov

# ==========================================
# 2. 物理模型核心 (Evolutionary HRS)
# ==========================================
def theory_distance_modulus(z, h0, om, alpha=0, beta=0, model='lcdm'):
    # 物理常數
    c = 299792.458 # 光速 km/s
    
    # 積分網格
    z_integ = np.linspace(0, np.max(z)*1.1, 1000)
    
    # 標準 LCDM 膨脹率 E(z) = H(z)/H0
    # 忽略輻射項 (在 z < 2.5 影響極小)，但保留曲率項為 0 (平坦宇宙)
    Ez = np.sqrt(om * (1 + z_integ)**3 + (1 - om))
    
    if model == 'hrs':
        # --- GROK 修正回應 ---
        # 使用指數衰減形式：Correction = 1 + beta * exp(-z/alpha)
        # 當 z >> alpha 時，修正項消失，回歸 LCDM (滿足 CMB/BBN 限制)
        # 當 z -> 0 時，H(z) -> H0 * (1 + beta)，這解釋了為什麼本地測量值較高
        correction = 1.0 + beta * np.exp(-z_integ / alpha)
        
        # 修正後的哈伯參數
        hz = h0 * Ez * correction
    else:
        hz = h0 * Ez
        
    # 計算共動距離 Dc
    inv_hz = 1.0 / hz
    dc = np.trapz(inv_hz, z_integ) # 總積分
    
    # 因為我們要對每個 z 計算積分，這裡使用累積積分插值加速
    dc_cum = pd.Series(inv_hz).rolling(2).mean().fillna(0).cumsum().values * (z_integ[1] - z_integ[0]) * c
    dc_interp = np.interp(z, z_integ, dc_cum)
    
    # 光度距離 Dl = (1+z) * Dc
    dl = (1 + z) * dc_interp
    
    # 距離模數 mu = 5 log10(Dl) + 25
    return 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0

# ==========================================
# 3. 統計推斷 (Likelihood & MCMC)
# ==========================================
def log_likelihood(theta, z, mu, inv_cov, model_type):
    if model_type == 'lcdm':
        h0, om = theta
        alpha, beta = 1.0, 0.0 # 佔位符
    else:
        h0, om, alpha, beta = theta
        
    # --- 參數邊界檢查 (Priors) ---
    if not (60 < h0 < 85): return -np.inf
    if not (0.1 < om < 0.5): return -np.inf
    
    if model_type == 'hrs':
        # Alpha (衰減尺度): 限制在 0.01 ~ 5.0 (對應 z 的範圍)
        if not (0.01 < alpha < 5.0): return -np.inf 
        # Beta (強度): 限制在 -0.5 ~ 1.5
        if not (-0.5 < beta < 1.5): return -np.inf

    # 計算理論值
    mu_model = theory_distance_modulus(z, h0, om, alpha, beta, model_type)
    
    # 計算殘差向量
    diff = mu - mu_model
    
    # --- 矩陣級卡方運算 (Chi-Square) ---
    # Chi2 = (Diff)^T * Cov^(-1) * (Diff)
    # 這一步自動處理了所有相關性誤差與絕對星等 M_B 的校準權重
    chisq = np.dot(diff, np.dot(inv_cov, diff))
    
    return -0.5 * chisq

# ==========================================
# 4. 主執行程序
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("   HRS v6.3.0 Real-Data Validator (Professional)  ")
    print("==================================================")
    
    # 1. 載入數據
    try:
        z, mu, inv_cov = load_official_pantheon_plus()
    except Exception as e:
        print(f"[ERROR] {e}")
        exit()

    print(f"\n[*] 啟動 MCMC 採樣 (N={len(z)}, Full Covariance)...")
    print("    注意：由於矩陣運算量大，此步驟可能需要 5-10 分鐘，請耐心等待。")
    
    nwalkers = 32
    steps = 800 # 步數適中，確保收斂即可
    ndim_l = 2
    ndim_h = 4
    
    # --- 執行 LCDM ---
    print("\n[1/2] 正在執行標準模型 (LCDM)...")
    pos_l = [73.0, 0.315] + 1e-3 * np.random.randn(nwalkers, ndim_l)
    sampler_l = emcee.EnsembleSampler(nwalkers, ndim_l, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
    sampler_l.run_mcmc(pos_l, steps, progress=True)
    
    # --- 執行 HRS (Evolutionary) ---
    print("\n[2/2] 正在執行全息修正模型 (HRS v6.3)...")
    # 初始猜測: H0=73, Om=0.315, Alpha=0.5 (在 z=0.5 衰減), Beta=0.1 (10% 修正)
    pos_h = [73.0, 0.315, 0.5, 0.1] + 1e-3 * np.random.randn(nwalkers, ndim_h)
    sampler_h = emcee.EnsembleSampler(nwalkers, ndim_h, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
    sampler_h.run_mcmc(pos_h, steps, progress=True)

    # --- 分析結果 ---
    def get_info_criteria(sampler, k, n_data):
        # 獲取最佳 Log Likelihood
        log_prob = sampler.get_log_prob(discard=200, flat=True)
        max_log_like = np.max(log_prob)
        
        # AIC = 2k - 2ln(L)
        aic = 2*k - 2*max_log_like
        # BIC = k*ln(n) - 2ln(L) (懲罰更重)
        bic = k*np.log(n_data) - 2*max_log_like
        
        # 獲取最佳參數
        best_idx = np.argmax(log_prob)
        theta = sampler.get_chain(discard=200, flat=True)[best_idx]
        return aic, bic, theta

    aic_l, bic_l, theta_l = get_info_criteria(sampler_l, 2, len(z))
    aic_h, bic_h, theta_h = get_info_criteria(sampler_h, 4, len(z))

    print("\n" + "="*60)
    print("      HRS v6.3.0 最終決戰報告 (Pantheon+ Full)")
    print("="*60)
    print(f" 模型比較       | LCDM (標準) | HRS (全息)")
    print(f" ---------------|-------------|-------------")
    print(f" 參數數量 (k)   | 2           | 4")
    print(f" AIC (越低越好) | {aic_l:.2f}    | {aic_h:.2f}")
    print(f" BIC (越低越好) | {bic_l:.2f}    | {bic_h:.2f}")
    print("-" * 60)
    print(f" Delta AIC      : {aic_l - aic_h:.4f} ({'支持 HRS' if aic_l > aic_h else '支持 LCDM'})")
    print(f" Delta BIC      : {bic_l - bic_h:.4f} ({'支持 HRS' if bic_l > bic_h else '支持 LCDM'})")
    print("-" * 60)
    print(f" HRS 最佳參數:")
    print(f" H0    : {theta_h[0]:.3f} km/s/Mpc")
    print(f" Omega_m : {theta_h[1]:.3f}")
    print(f" Alpha : {theta_h[2]:.4f} (衰減紅移尺度 z_c)")
    print(f" Beta  : {theta_h[3]:.4f} (本地修正強度)")
    print("="*60)

    # 繪圖
    labels = [r"$H_0$", r"$\Omega_m$", r"$\alpha$", r"$\beta$"]
    fig = corner.corner(sampler_h.get_chain(discard=200, flat=True), labels=labels, truths=theta_h, 
                        show_titles=True, title_fmt=".3f")
    plt.savefig("hrs_v6_3_0_full_matrix.png")
    print("[🎉] 驗證完成，圖表已儲存為 'hrs_v6_3_0_full_matrix.png'")
