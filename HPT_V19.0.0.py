import subprocess
import sys

# --- 自動環境檢查機制 ---
def setup_environment():
    required = {"numpy", "pandas", "matplotlib", "scipy", "requests", "emcee", "corner", "torch"}
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
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# ==========================================
# v19.0 HIA: 共形資訊度規重構 (Metric Reconstruction)
# ==========================================

class ConformalMetricEngine:
    def __init__(self):
        # 宇宙學背景 (Planck 2018)
        self.H0_bg = 67.4 
        self.Om = 0.315
        self.Ol = 0.685
        self.c = 299792.458 # 光速 km/s
        
        # v19.0 幾何參數 (新憲法)
        # 我們需要讓空洞內的距離縮短約 8% (67.4 / 73.04 ≈ 0.92)
        # scaling_strength 控制縮放的強度
        self.scaling_strength = 0.085 
        
        # KBC 空洞參數
        self.void_z_edge = 0.07 # 空洞邊界
        self.void_sharpness = 15.0 # 邊緣過渡的平滑度

    def get_scaling_factor(self, z):
        """
        [共形縮放因子 eta(z)]
        這是 v19.0 的靈魂。它定義了時空幾何如何隨環境改變。
        在空洞內 (low z)，eta > 1 (距離被壓縮，H0 變大)。
        在深空 (high z)，eta -> 1 (回歸 GR)。
        """
        # 使用 Sigmoid 函數模擬從空洞到深空的平滑過渡
        # 當 z 小時，exp 為大正數，sigmoid -> 1，factor -> 1 + strength
        # 當 z 大時，exp 為 0，sigmoid -> 0，factor -> 1
        transition = 1.0 / (1.0 + np.exp((z - self.void_z_edge) * self.void_sharpness))
        
        eta = 1.0 + self.scaling_strength * transition
        return eta

    def compute_observable_universe(self, z_array):
        # 1. 計算背景 LCDM 的光度距離 D_L_bg
        # E(z) = H(z)/H0
        E_z = np.sqrt(self.Om * (1 + z_array)**3 + self.Ol)
        
        # 積分 comoving distance: Dc = c/H0 * int(1/E(z) dz)
        # 注意: 這裡做數值積分
        inv_E = 1.0 / E_z
        Dc_integral = cumtrapz(inv_E, z_array, initial=0)
        Dc = (self.c / self.H0_bg) * Dc_integral
        
        # 背景光度距離 Dl = (1+z) * Dc
        Dl_bg = (1 + z_array) * Dc
        
        # 2. 應用 v19.0 共形縮放
        # HIA 假設：真實測量的距離被資訊場縮放了
        scaling = self.get_scaling_factor(z_array)
        Dl_obs = Dl_bg / scaling  # 距離變短 -> H0 變大
        
        # 3. 反推 "Apparent H0" (觀測到的哈伯常數)
        # 根據哈伯定律: H0 ~ c * z / Dl (低紅移近似)
        # 為了精確，我們用 H_obs(z) = c * z / (Dl_obs / (1+z)) 
        # 但更直觀的是看 Distance Modulus 殘差，這裡我們先輸出 Apparent H0
        
        # 避免 z=0 除以零，從索引 1 開始計算
        H0_apparent = np.zeros_like(z_array)
        with np.errstate(divide='ignore', invalid='ignore'):
            # 使用近似公式 H0 = v / d = (c*z) / (Dl / (1+z)) ? 
            # 嚴格來說，我們定義 H0_apparent = H0_bg * scaling
            # 因為 Dl_obs = Dl_bg / scaling => H_obs ~ H_bg * scaling
            H0_apparent = self.H0_bg * scaling
            
        return z_array, H0_apparent, Dl_obs, Dl_bg

# ==========================================
# 執行 v19.0 全紅移掃描
# ==========================================
def run_v19_reconstruction():
    print("--- [v19.0 HIA: 度規重構與平滑驗證] ---")
    
    engine = ConformalMetricEngine()
    z_vals = np.linspace(0.001, 1.5, 500)
    
    z, H_app, Dl_obs, Dl_bg = engine.compute_observable_universe(z_vals)
    
    # --- 關鍵點提取 ---
    idx_local = np.abs(z - 0.01).argmin()
    idx_edge = np.abs(z - 0.07).argmin()
    idx_deep = np.abs(z - 1.0).argmin()
    
    print(f"Local Apparent H0 (z=0.01): {H_app[idx_local]:.2f} km/s/Mpc (Target: 73.04)")
    print(f"Void Edge H0      (z=0.07): {H_app[idx_edge]:.2f} km/s/Mpc (平滑過渡檢測)")
    print(f"Deep Space H0     (z=1.00): {H_app[idx_deep]:.2f} km/s/Mpc (Target: 67.4)")
    
    # --- 繪圖 ---
    plt.figure(figsize=(10, 6))
    
    # 繪製 H0 視在值
    plt.plot(z, H_app, 'r-', lw=3, label='v19.0 Apparent H0 (Information Metric)')
    plt.axhline(73.04, color='green', ls=':', label='SH0ES (Local) Target')
    plt.axhline(67.4, color='black', ls='--', label='Planck (CMB) Baseline')
    
    # 標示空洞區域
    plt.axvline(0.07, color='gray', alpha=0.5)
    plt.text(0.02, 70, "KBC Void\n(Information Lensing)", color='red', fontsize=10)
    plt.text(0.5, 68, "Cosmic Average\n(Standard Metric)", color='black', fontsize=10)
    
    plt.title('v19.0 HIA: Conformal Metric Reconstruction')
    plt.xlabel('Redshift z')
    plt.ylabel('Apparent Hubble Constant H0 [km/s/Mpc]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(66, 75)
    plt.xlim(0, 0.2) # 聚焦在近場過渡區查看是否平滑
    plt.show()

if __name__ == "__main__":
    run_v19_reconstruction()

