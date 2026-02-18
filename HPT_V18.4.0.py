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

# ==========================================
# v18.4 Reality Check: 套用 2026 KBC Void 真實觀測參數
# ==========================================

class RealityValidator:
    def __init__(self):
        self.H0_cosmic = 67.4
        # 根據 2026 最新文獻：KBC Void 在 300Mpc 內的密度虧損約為 46%
        self.real_world_delta = -0.46 
        
        # 保持 v18.3 的 HIA 核心參數
        self.alpha = 1.2
        self.I_crit = 0.6
        self.n_power = 3.5

    def get_real_density(self, z):
        # 結構隨紅移縮減，且在局部 z<0.07 處存在真實空洞
        base_growth = 1.0 / (1.0 + z)**2
        # 模擬一個半徑約 300Mpc (z~0.07) 的真實空洞剖面
        void_effect = self.real_world_delta if z < 0.07 else 0.0
        return base_growth * (1.0 + void_effect)

    def run_validation(self):
        # 計算 z=0.01 (SH0ES 觀測區) 的預測值
        z_obs = 0.01
        dz = 0.001
        
        # 背景 H(z)
        E_z = np.sqrt(0.315 * (1 + z_obs)**3 + 0.685)
        H_bg = self.H0_cosmic * E_z
        
        # HIA 修正
        I_now = self.get_real_density(z_obs)
        I_next = self.get_real_density(z_obs + dz)
        
        # 耦合函數 F
        F = lambda I: 1.0 + self.alpha * (I**self.n_power / (I**self.n_power + self.I_crit**self.n_power))
        
        F_now = F(I_now)
        F_next = F(I_next)
        dF_dt = ((F_next - F_now) / dz) * (-H_bg * (1 + z_obs))
        
        H_predict = H_bg + 0.5 * (dF_dt / F_now)
        
        print(f"--- [v18.4 現實數據驗證報告] ---")
        print(f"輸入真實空洞密度 ($\delta$): {self.real_world_delta * 100:.1f}%")
        print(f"HIA 預測之局部 H0: {H_predict:.2f} km/s/Mpc")
        print(f"現實 SH0ES 觀測值: 73.04 km/s/Mpc")
        print(f"偏差值 (Residual): {abs(H_predict - 73.04):.2f}")
        
        if abs(H_predict - 73.04) < 1.0:
            print("\n>> [結果]: 通過現實考驗！模型與真實觀測誤差在 1.0 以內。")
        else:
            print("\n>> [結果]: 遭到現實擊碎。模型修正力不足或過頭。")

validator = RealityValidator()
validator.run_validation()

