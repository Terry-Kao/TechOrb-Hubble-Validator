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
from scipy.optimize import fsolve

class HIAFinalValidator:
    def __init__(self):
        self.H0_bg = 67.4 
        self.H0_target = 73.04
        self.Om = 0.315
        self.Ol = 0.685
        
        # 初始猜測
        self.void_z_edge = 0.08  # 稍微擴大空洞核心區，確保 z=0.01 處於平台期
        self.void_sharpness = 20.0 

    def get_eta(self, z, strength):
        # v19.0 共形縮放因子公式
        transition = 1.0 / (1.0 + np.exp((z - self.void_z_edge) * self.void_sharpness))
        return 1.0 + strength * transition

    def find_best_strength(self):
        # 尋找能讓 H0(0.01) = 73.04 的 strength
        def objective(s):
            return self.H0_bg * self.get_eta(0.01, s) - self.H0_target
        
        best_s = fsolve(objective, 0.08)[0]
        return best_s

    def mu(self, Dl):
        # 距離模數公式: mu = 5 * log10(Dl) + 25 (Dl 單位為 Mpc)
        # 避免 Dl 為 0
        return 5 * np.log10(np.maximum(Dl, 1e-10)) + 25

    def run_analysis(self):
        strength = self.find_best_strength()
        z_vals = np.linspace(0.001, 1.0, 500)
        c = 299792.458
        
        # 1. 計算背景距離 (Planck)
        # 近似計算：Dl = c/H0 * z * (1 + 0.5*(1-q0)*z) 
        # 為求精確，我們使用 z 積分
        from scipy.integrate import cumtrapz
        E_z = np.sqrt(self.Om * (1 + z_vals)**3 + self.Ol)
        Dc = (c / self.H0_bg) * cumtrapz(1.0/E_z, z_vals, initial=0)
        Dl_bg = (1 + z_vals) * Dc
        
        # 2. HIA 修正後的距離 (觀測距離)
        eta = self.get_eta(z_vals, strength)
        Dl_obs = Dl_bg / eta
        
        # 3. 視在哈伯常數
        H0_app = self.H0_bg * eta
        
        # 4. 殘差分析: Delta_mu = mu(HIA) - mu(Planck LCDM)
        # 如果 Delta_mu < 0，代表超新星看起來比預期更亮（即距離更近）
        delta_mu = self.mu(Dl_obs) - self.mu(Dl_bg)
        
        print(f"--- [v19.1 HIA 終極校準報告] ---")
        print(f"鎖定強度 (Scaling Strength): {strength:.5f}")
        print(f"空洞邊界 (Void Edge z): {self.void_z_edge}")
        print(f"預測 z=0.01 H0: {H0_app[np.abs(z_vals-0.01).argmin()]:.4f}")
        print(f"深空 z=1.00 H0: {H0_app[-1]:.4f}")
        
        # --- 繪圖 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 圖1: H0 演化
        ax1.plot(z_vals, H0_app, 'r-', lw=2, label='v19.1 HIA Apparent H0')
        ax1.axhline(73.04, color='green', ls=':', label='SH0ES Target')
        ax1.axhline(67.4, color='black', ls='--', label='Planck Baseline')
        ax1.set_ylabel('Apparent H0 (km/s/Mpc)')
        ax1.set_title('v19.1: Precision Lock & Residual Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # 圖2: 殘差圖 Delta_mu (SNIa 指紋)
        ax2.plot(z_vals, delta_mu, 'b-', lw=2, label='$\Delta\mu$ (HIA - Planck)')
        ax2.set_ylabel('Distance Modulus Residual $\Delta\mu$')
        ax2.set_xlabel('Redshift z')
        ax2.axhline(0, color='black', lw=1)
        # 標註：這就是超新星數據觀察到的「超量亮度」區域
        ax2.fill_between(z_vals, 0, delta_mu, where=(z_vals < self.void_z_edge), color='blue', alpha=0.1)
        ax2.text(0.01, -0.05, "Supernovae look brighter\nin the Void", color='blue')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.show()

validator = HIAFinalValidator()
validator.run_analysis()

