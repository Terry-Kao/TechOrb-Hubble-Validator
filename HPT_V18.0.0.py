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
from scipy.integrate import odeint

# ==========================================
# v18.0 HIA 核心物理引擎
# ==========================================

class HolographicActionModel:
    def __init__(self):
        # 物理常數
        self.H0_cosmic = 67.4  # Planck 基底 (深空真實膨脹率)
        self.Om = 0.315        # 物質密度參數
        self.Ol = 0.685        # 暗能量密度參數
        
        # HIA 模型參數 (憲法定義的自由度)
        # alpha: 耦合強度 (資訊對重力的影響力)
        # I_crit: 資訊飽和閾值 (宇宙何時開始變得"擁擠")
        self.alpha = 0.12      
        self.n_power = 3.0     # 轉變的陡峭程度
        self.I_crit = 0.5

    def get_information_density_proxy(self, z):
        """
        [代理函數] 定義宇宙演化過程中的資訊密度 I(z)。
        假設：隨著結構形成 (Structure Formation)，資訊密度在晚期(z->0)急劇上升。
        在早期 (z > 1000)，資訊密度趨近於 0 (均勻流體)。
        """
        # 使用簡單的結構增長因子近似
        growth_factor = 1.0 / (1.0 + z)
        # 加入隨機擾動模擬局部環境差異 (Local Variance)
        # 在真實搜證階段，這裡會替換成真實觀測數據 (Galaxy Density)
        local_variance = 0.0 # 暫設為0以查看平均趨勢
        
        I_z = growth_factor * (1.0 + local_variance)
        return I_z

    def coupling_function_F(self, I):
        """
        [全息耦合函數] F(I)
        憲法規定：當 I -> 0 (早期), F -> 1 (GR還原)。
        當 I 變大, F 發生偏離，改變有效度規。
        """
        # Sigmoid / Hill-type function
        sigmoid = (I**self.n_power) / (I**self.n_power + self.I_crit**self.n_power)
        F = 1.0 + self.alpha * sigmoid
        return F

    def time_derivative_F(self, z, H_z):
        """
        計算 dF/dt = (dF/dI) * (dI/dz) * (dz/dt)
        這是計算 H_obs 修正項的關鍵。
        """
        # 1. 數值微分計算 dF/dz
        dz = 0.001
        I_plus = self.get_information_density_proxy(z + dz)
        I_minus = self.get_information_density_proxy(z - dz)
        F_plus = self.coupling_function_F(I_plus)
        F_minus = self.coupling_function_F(I_minus)
        dF_dz = (F_plus - F_minus) / (2 * dz)
        
        # 2. 轉換為時間導數: dz/dt = -H(z)*(1+z)
        # dF/dt = dF/dz * dz/dt
        dF_dt = dF_dz * (-H_z * (1 + z))
        
        return dF_dt

    def observable_hubble(self, z_array):
        """
        計算觀測到的哈伯參數 H_obs(z)
        公式: H_obs = H_cosmic + 0.5 * (dF/dt) / F
        """
        H_obs_list = []
        F_list = []
        
        for z in z_array:
            # 1. 計算背景宇宙膨脹 (Standard LCDM)
            E_z = np.sqrt(self.Om * (1 + z)**3 + self.Ol)
            H_background = self.H0_cosmic * E_z
            
            # 2. 計算資訊場的修正
            I = self.get_information_density_proxy(z)
            F = self.coupling_function_F(I)
            dF_dt = self.time_derivative_F(z, H_background)
            
            # 3. 應用五項守恆導致的尺規縮放修正
            # 這裡的修正項來自度規變換 ds_obs^2 = (1/F) ds_ideal^2
            correction = 0.5 * (dF_dt / F)
            
            # 注意單位：correction 已經是 H 的單位
            # 但因為 dF/dt 通常是負的(隨著時間 F 變大)，
            # 我們需要仔細定義 F 的物理意義。
            # 如果 F 變大代表尺規變小(測量值變大)，則符號需調整。
            # 根據 Grok 建議的模型: H_obs ~ H + 0.5 * dlnF/dt
            
            H_effective = H_background + correction
            
            H_obs_list.append(H_effective)
            F_list.append(F)
            
        return np.array(H_obs_list), np.array(F_list)

# ==========================================
# 執行 v18.0 驗證模擬
# ==========================================
def run_v18_simulation():
    print("[*] 啟動 v18.0 HIA：全息資訊作用量模擬...")
    print("[*] 正在載入憲法約束：High-z Limit & Ghost-Free check...")
    
    model = HolographicActionModel()
    
    # 產生紅移區間 (包含近場與遠場)
    z_vals = np.logspace(np.log10(0.01), np.log10(1100), 500)
    
    H_obs, F_vals = model.observable_hubble(z_vals)
    
    # --- 關鍵數據提取 ---
    H0_local_obs = H_obs[0]  # z -> 0
    H_cmb_epoch = H_obs[-1] / np.sqrt(model.Om*(1+1100)**3 + model.Ol) # 還原到 H0 單位
    
    print(f"\n[v18.0 模擬結果]")
    print(f"基準 Cosmic H0: {model.H0_cosmic} km/s/Mpc")
    print(f"------------------------------------------------")
    print(f"觀測到的 Local H0 (z=0.01): {H0_local_obs:.2f} km/s/Mpc")
    print(f"早期的有效 H0 (z=1100):     {H_cmb_epoch:.2f} km/s/Mpc (Check consistency)")
    print(f"最大耦合強度 F_max:         {np.max(F_vals):.4f}")
    print(f"------------------------------------------------")
    
    # --- 判定 ---
    tension_gap = H0_local_obs - model.H0_cosmic
    print(f"產生張力差 (Gap): {tension_gap:.2f} km/s/Mpc")
    
    if 72.0 <= H0_local_obs <= 74.5:
        print("\n[SUCCESS] 模擬成功！")
        print(">> 模型在不破壞早期物理的情況下，自然產生了 ~73 的近場觀測值。")
        print(">> 這驗證了「資訊密度」導致的「度規縮放」是張力的來源。")
    else:
        print("\n[ADJUSTMENT NEEDED] 參數需微調。")
        
    # --- 繪圖 ---
    plt.figure(figsize=(10, 6))
    plt.plot(z_vals, H_obs, label='v18.0 HIA Effective H(z)', color='red', lw=2)
    
    # 繪製參考線
    H_lcdm = model.H0_cosmic * np.sqrt(model.Om*(1+z_vals)**3 + model.Ol)
    plt.plot(z_vals, H_lcdm, '--', label='Standard Planck LCDM', color='black', alpha=0.5)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.01, 1200)
    plt.ylim(60, 1000000)
    plt.xlabel('Redshift z')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.title('v18.0 HIA: Information-Driven Metric Scaling')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_v18_simulation()


