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
from scipy.integrate import quad

class HIA_Engine:
    """HIA v24.1: Holographic Information Alignment Engine"""
    def __init__(self, alpha=0.0682, z_edge=0.5, h0_base=67.4, om=0.315):
        self.alpha = alpha
        self.z_edge = z_edge
        self.h0_base = h0_base
        self.om = om
        self.ob = 0.049
        self.og = 5.4e-5
        self.z_star = 1089.92
        self.theta_star_target = 0.010411 # Planck 2018 基準

    def get_h_z(self, z):
        # 1. 標準 Planck 2018 擴張基底
        h_lcdm = self.h0_base * np.sqrt(self.om*(1+z)**3 + self.ob*(1+z)**3 + self.og*(1+z)**4 + (1-self.om-self.ob-self.og))
        # 2. HIA 局域高斯屏蔽增益 (全息相變釋放)
        gain = 1 + self.alpha * np.exp(-(z / self.z_edge)**2)
        return h_lcdm * gain

    def validate_theta_star(self):
        # 計算早期宇宙聲學視界 rs
        def rs_integrand(z):
            R = (3.0 * self.ob) / (4.0 * self.og * (1+z))
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return cs / self.get_h_z(z)
        
        rs, _ = quad(rs_integrand, self.z_star, 1e7)
        
        # 計算到最後散射面的共動距離 DA
        da, _ = quad(lambda z: 1.0 / self.get_h_z(z), 0, self.z_star)
        
        theta_hia = rs / da
        precision = abs(theta_hia - self.theta_star_target) / self.theta_star_target
        
        return theta_hia, precision

# ==========================================
# 本地端執行測試 (當直接執行此腳本時會觸發)
# ==========================================
if __name__ == "__main__":
    print("=== HIA v24.1 Engine Initialization ===")
    engine = HIA_Engine()
    
    # 測試 1: 局域宇宙的 Hubble 常數 (z=0)
    h0_local = engine.get_h_z(0)
    print(f"[Local Universe] H(z=0) = {h0_local:.2f} km/s/Mpc (Target: 72.00)")
    
    # 測試 2: 早期宇宙的 Hubble 參數 (z=1089)
    # 證明高斯屏蔽在早期宇宙完美失效，回歸標準模型
    h_1089_hia = engine.get_h_z(1089)
    engine_lcdm = HIA_Engine(alpha=0) # 關閉 HIA 效應的純 LCDM 基準
    h_1089_lcdm = engine_lcdm.get_h_z(1089)
    print(f"[Early Universe] H(z=1089) HIA  = {h_1089_hia:.2f}")
    print(f"[Early Universe] H(z=1089) LCDM = {h_1089_lcdm:.2f}")
    
    # 測試 3: CMB 聲學尺度幾何壓力測試
    print("\n=== Running CMB Theta* Stress Test ===")
    theta, prec = engine.validate_theta_star()
    print(f"HIA 100*Theta_* : {theta*100:.6f}")
    print(f"Planck 100*Theta_*: {engine.theta_star_target*100:.6f}")
    print(f"Relative Precision: {prec:.2e}")
    
    if prec < 1e-4:
        print("\n✅ SUCCESS: Geometric tension resolved. CMB Horizon is strictly preserved.")
    else:
        print("\n❌ WARNING: Precision requirement not met.")

