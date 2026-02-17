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
import os
import scipy.linalg as la

# ==========================================
# 1. 穩健型數據載入 (含正規化處理)
# ==========================================
def load_data_calm():
    print("[*] v6.3.3 冷靜版啟動：正在載入 Pantheon+ 數據...")
    
    dat_file = "Pantheon+SH0ES.dat"
    cov_file = "Pantheon+SH0ES_STAT+SYS.cov"
    
    if not (os.path.exists(dat_file) and os.path.exists(cov_file)):
        print("[❌] 錯誤：找不到數據檔案，請確認檔案已上傳。")
        return None, None, None, None

    # 讀取數據
    df = pd.read_csv(dat_file, sep=r'\s+')
    
    # 讀取協方差矩陣 (處理標頭)
    raw_data = np.fromfile(cov_file, sep=' ')
    n_header = int(raw_data[0])
    matrix_data = raw_data[1:]
    
    if len(matrix_data) != n_header**2:
        print("[!] 標頭格式異常，嘗試直接讀取...")
        matrix_data = raw_data # Fallback
        n_header = int(np.sqrt(len(raw_data)))

    cov_matrix = matrix_data.reshape((n_header, n_header))

    # 數據篩選 (z > 0.01 宇宙學標準)
    mask = (df['zHD'] > 0.01)
    df_clean = df[mask].reset_index(drop=True)
    z_obs = df_clean['zHD'].values
    mu_obs = df_clean['m_b_corr'].values
    
    # 矩陣切割
    indices = df.index[mask].values
    cov_cut = cov_matrix[np.ix_(indices, indices)]
    
    # --- [關鍵修正 1] 矩陣正規化 ---
    # 對角線加入微小擾動，防止奇異值導致 BIC 爆炸
    print("    -> 正在執行矩陣正規化 (Regularization)...")
    cov_cut += np.eye(len(cov_cut)) * 1e-5
    
    # 使用 Cholesky 分解求反矩陣 (比 np.linalg.inv 更穩定)
    try:
        # C = L * L.T => C^-1 = (L^-1).T * L^-1
        L = np.linalg.cholesky(cov_cut)
        inv_L = np.linalg.inv(L)
        inv_cov = np.dot(inv_L.T, inv_L)
    except np.linalg.LinAlgError:
        print("    [!] Cholesky 分解失敗，退回使用偽逆矩陣 (Pinverse)...")
        inv_cov = np.linalg.pinv(cov_cut)
        
    print(f"[✅] 數據準備完成：{len(z_obs)} 點，矩陣數值已穩定化。")
    return z_obs, mu_obs, inv_cov, cov_cut

# ==========================================
# 2. 物理模型 (幾何形狀預測)
# ==========================================
def theory_mu_shape(z, om, alpha, beta, model='lcdm'):
    # 這裡我們不傳入 H0，因為 H0 只是一個垂直位移
    # 我們計算的是 "形狀" (Shape)，位移由解析解處理
    
    c = 299792.458
    # 積分
    z_integ = np.linspace(0, np.max(z)*1.05, 1000)
    Ez = np.sqrt(om * (1 + z_integ)**3 + (1 - om))
    
    if model == 'hrs':
        # 指數衰減全息修正
        correction = 1.0 + beta * np.exp(-z_integ / alpha)
        hz = Ez * correction # 這裡沒有 H0，因為它是相對變化
    else:
        hz = Ez
        
    inv_hz = 1.0 / hz
    dc_cum = np.cumsum(inv_hz) * (z_integ[1] - z_integ[0]) * c
    dc_interp = np.interp(z, z_integ, dc_cum)
    dl = (1 + z) * dc_interp
    
    # 這裡的 mu = 5 log10(Dl) + 25
    # 我們先算一個 "基礎 mu" (假設 H0=100)
    mu_base = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0
    return mu_base

# ==========================================
# 3. 似然函數 (含解析邊際化修正)
# ==========================================
def log_likelihood(theta, z, mu, inv_cov, model_type):
    # --- 參數拆解 ---
    if model_type == 'lcdm':
        om = theta[0]
        alpha = 1.0; beta = 0.0
    else:
        om, alpha, beta = theta
    
    # --- [關鍵修正 2] 寬廣的參數先驗 (No Walls) ---
    if not (0.0 < om < 1.0): return -np.inf
    if model_type == 'hrs' and not (0.01 < alpha < 20.0 and -5.0 < beta < 5.0): return -np.inf

    # --- 理論預測 (Shape only) ---
    mu_model = theory_mu_shape(z, om, alpha, beta, model_type)
    
    # --- [關鍵修正 3] 解析邊際化 (Analytical Marginalization) ---
    # 我們不擬合 H0 與 M，而是用矩陣公式自動求出最佳位移
    # 這是消除 "數值偏移作弊" 的唯一方法
    
    diff = mu - mu_model
    
    # 向量計算權重
    W = np.sum(inv_cov)  # 權重總和
    if W == 0: return -np.inf
    
    weighted_diff = np.sum(np.dot(inv_cov, diff)) # 加權偏移
    delta = weighted_diff / W # 這是最佳的 (H0 + M) 位移值
    
    # 修正後的殘差
    diff_corr = diff - delta
    
    # 計算 Chi^2
    chisq = np.dot(diff_corr, np.dot(inv_cov, diff_corr))
    
    return -0.5 * chisq

# ==========================================
# 4. 執行分析
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("   HRS v6.3.3 Calm Edition (No Artifacts)")
    print("==========================================")
    
    z, mu, inv_cov, cov = load_data_calm()
    
    if z is not None:
        nwalkers, steps = 32, 1200
        
        # --- 執行 LCDM (1 參數: Omega_m) ---
        print("\n[*] 執行 LCDM (基準模型)...")
        # 初始值: Om=0.3
        sampler_l = emcee.EnsembleSampler(nwalkers, 1, log_likelihood, args=(z, mu, inv_cov, 'lcdm'))
        sampler_l.run_mcmc(0.3 + 1e-3*np.random.randn(nwalkers, 1), steps, progress=True)
        
        # --- 執行 HRS (3 參數: Om, Alpha, Beta) ---
        print("\n[*] 執行 HRS (全息修正)...")
        # 初始值: Om=0.3, Alpha=1.0, Beta=0.1
        pos_h = [0.3, 1.0, 0.1] + 1e-3*np.random.randn(nwalkers, 3)
        sampler_h = emcee.EnsembleSampler(nwalkers, 3, log_likelihood, args=(z, mu, inv_cov, 'hrs'))
        sampler_h.run_mcmc(pos_h, steps, progress=True)

        # --- 真實統計 ---
        def get_bic(sampler, k, n_data):
            lp = sampler.get_log_prob(discard=300, flat=True)
            max_log_like = np.max(lp)
            # BIC = k*ln(n) - 2*ln(L)
            return k * np.log(n_data) - 2*max_log_like, sampler.get_chain(discard=300, flat=True)[np.argmax(lp)]

        bic_l, theta_l = get_bic(sampler_l, 1, len(z)) # k=1 (Om only)
        bic_h, theta_h = get_bic(sampler_h, 3, len(z)) # k=3 (Om, Alpha, Beta)
        
        delta_bic = bic_l - bic_h # 正值表示支持 HRS

        print("\n" + "="*50)
        print("   HRS v6.3.3 真實數據分析報告")
        print("="*50)
        print(f" Delta BIC (真實): {delta_bic:.4f}")
        print("-" * 50)
        print(f" HRS 最佳參數:")
        print(f" Omega_m : {theta_h[0]:.4f}")
        print(f" Alpha   : {theta_h[1]:.4f}")
        print(f" Beta    : {theta_h[2]:.4f}")
        print("-" * 50)
        
        if delta_bic > 6:
            print(" [結論] 強烈支持 (Strong Evidence, Delta BIC > 6)")
        elif delta_bic > 2:
            print(" [結論] 正面支持 (Positive Evidence, Delta BIC > 2)")
        elif delta_bic > -2:
            print(" [結論] 兩者無法區分 (Inconclusive)")
        else:
            print(" [結論] 支持標準模型 (Favor LCDM)")
            
        print("="*50)
        
        # 繪圖
        labels = [r"$\Omega_m$", r"$\alpha$", r"$\beta$"]
        fig = corner.corner(sampler_h.get_chain(discard=300, flat=True), labels=labels, truths=theta_h, show_titles=True)
        plt.savefig("hrs_v6_3_3_calm.png")
        print("[🎉] 分析完成。")

