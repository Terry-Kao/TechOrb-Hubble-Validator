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

# ==========================================
# v18.6 HIA: 全紅移覆蓋與殘差分析引擎
# ==========================================

class FullRangeValidator:
    def __init__(self):
        # 宇宙學參數 (Planck 2018)
        self.H0_cosmic = 67.4 
        self.Om = 0.315
        self.Ol = 0.685
        
        # v18.5 鎖定的物理常數 (我們的"憲法")
        self.alpha = 0.4122  # 最大修正力
        self.beta = 0.4102   # 響應靈敏度
        
        # 現實環境參數
        self.void_depth = -0.46 # KBC 空洞內部密度虧損
        self.void_radius_z = 0.07 # 空洞邊界 (約 300 Mpc)

    def get_density_profile(self, z):
        """
        [真實視線模擬]
        我們從空洞中心往外看。
        在 z < 0.07 時，我們處於低密度區。
        在 z > 0.07 後，視線進入平均密度的宇宙深處。
        """
        base_growth = 1.0 / (1.0 + z)**2
        
        # 使用平滑過渡函數 (Sigmoid) 模擬空洞邊緣，避免物理斷層
        # transition: 0 (in void) -> 1 (outside)
        width = 0.02
        transition = 1.0 / (1.0 + np.exp(-(z - self.void_radius_z) / width))
        
        # 局部密度 delta 從 -0.46 過渡到 0.0
        current_delta = self.void_depth * (1.0 - transition)
        
        return base_growth * (1.0 + current_delta)

    def coupling_F(self, I):
        # v18.5 確立的 Tanh 飽和函數
        return 1.0 + self.alpha * np.tanh(self.beta * I)

    def calculate_hubble_track(self, z_array):
        H_hia = []
        H_lcdm = []
        
        for z in z_array:
            # 1. 標準 LCDM 基底
            E_z = np.sqrt(self.Om * (1 + z)**3 + self.Ol)
            H_bg = self.H0_cosmic * E_z
            H_lcdm.append(H_bg)
            
            # 2. HIA 修正
            dz = 0.001
            I_now = self.get_density_profile(z)
            I_next = self.get_density_profile(z + dz)
            
            F_now = self.coupling_F(I_now)
            F_next = self.coupling_F(I_next)
            
            # dF/dt 計算
            dF_dt = ((F_next - F_now) / dz) * (-H_bg * (1 + z))
            
            # HIA 預測值
            H_model = H_bg + 0.5 * (dF_dt / F_now)
            H_hia.append(H_model)
            
        return np.array(H_hia), np.array(H_lcdm)

# ==========================================
# 執行驗證與繪圖
# ==========================================
def run_full_scan():
    print("--- [v18.6 全紅移掃描啟動] ---")
    print("正在模擬視線穿過 KBC 空洞 (z < 0.07) 進入深空...")
    
    validator = FullRangeValidator()
    
    # 產生從近場到遠場的紅移點
    z_vals = np.linspace(0.005, 2.0, 200)
    H_hia, H_lcdm = validator.calculate_hubble_track(z_vals)
    
    # 計算殘差 (HIA - LCDM)
    residuals = H_hia - H_lcdm
    
    # --- 關鍵檢核點 ---
    idx_local = np.abs(z_vals - 0.01).argmin()
    idx_void_edge = np.abs(z_vals - 0.07).argmin()
    idx_far = np.abs(z_vals - 1.5).argmin()
    
    print(f"\n[關鍵位置檢核]")
    print(f"Local (z=0.01): {H_hia[idx_local]:.2f} (目標: ~73)")
    print(f"Void Edge (z=0.07): {H_hia[idx_void_edge]:.2f} (過渡區)")
    print(f"Deep Space (z=1.5): {H_hia[idx_far]:.2f} vs LCDM {H_lcdm[idx_far]:.2f}")
    
    # --- 繪圖 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 主圖: H(z)/(1+z) 為了視覺清晰度，通常除以 (1+z) 來平緩曲線
    ax1.plot(z_vals, H_hia / (1+z_vals), 'r-', lw=2.5, label='v18.6 HIA (Inhomogeneous)')
    ax1.plot(z_vals, H_lcdm / (1+z_vals), 'k--', lw=1.5, alpha=0.7, label='Standard LCDM (Planck)')
    
    # 標示空洞邊界
    ax1.axvline(x=0.07, color='gray', linestyle=':', label='KBC Void Edge')
    ax1.text(0.02, 68, "Local Void\n(High H0)", color='red', fontsize=9)
    ax1.text(0.2, 60, "Cosmic Average\n(Standard H0)", color='black', fontsize=9)
    
    ax1.set_ylabel('H(z) / (1+z) [km/s/Mpc]')
    ax1.set_title('v18.6 Full Redshift Validation: From Void to Cosmos')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 殘差圖
    ax2.plot(z_vals, residuals, 'r-', lw=1.5)
    ax2.axhline(0, color='k', linestyle='--')
    ax2.fill_between(z_vals, 0, residuals, color='red', alpha=0.1)
    ax2.set_ylabel('Diff (HIA - LCDM)')
    ax2.set_xlabel('Redshift z')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_scan()

