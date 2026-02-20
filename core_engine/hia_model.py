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
    """HIA v24.2: Rigorous Cosmological Background Engine"""
    def __init__(self, alpha=0.0682, z_edge=0.5, h0_base=67.4, om=0.315):
        self.alpha = alpha
        self.z_edge = z_edge
        self.h0_base = h0_base
        
        # 精確的 Planck 2018 宇宙學常數
        self.om = om          # 總物質密度 (Cold Dark Matter + Baryons)
        self.ob = 0.049       # 重子物質密度 (僅用於計算聲速)
        self.og = 5.38e-5     # 光子輻射密度
        self.on = 0.2271 * 3.046 * self.og # 中微子輻射密度 (N_eff = 3.046)
        self.orad = self.og + self.on      # 總輻射密度
        
        self.z_star = 1089.92
        self.theta_star_target = 0.010411 # Planck 2018 基準

    def get_h_z(self, z):
        # 1. 修正後的標準 Planck 2018 擴張基底 (包含輻射與暗能量)
        # 注意: (1 - om - orad) 是暗能量 Omega_Lambda
        E_z = np.sqrt(self.om * (1+z)**3 + self.orad * (1+z)**4 + (1 - self.om - self.orad))
        h_lcdm = self.h0_base * E_z
        
        # 2. HIA 局域高斯屏蔽增益
        gain = 1 + self.alpha * np.exp(-(z / self.z_edge)**2)
        return h_lcdm * gain

    def validate_theta_star(self):
        # 計算早期宇宙聲學視界 rs
        def rs_integrand(z):
            # R = 3 * rho_b / 4 * rho_gamma
            R = (3.0 * self.ob) / (4.0 * self.og * (1+z))
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return cs / self.get_h_z(z)
        
        # 積分從 z_star 到無限大 (以 1e6 代替)
        rs, _ = quad(rs_integrand, self.z_star, 1e6)
        
        # 計算到最後散射面的共動距離 DA
        da, _ = quad(lambda z: 1.0 / self.get_h_z(z), 0, self.z_star)
        
        theta_hia = rs / da
        precision = abs(theta_hia - self.theta_star_target) / self.theta_star_target
        
        return theta_hia, precision, rs, da

# ==========================================
# 本地端執行測試
# ==========================================
if __name__ == "__main__":
    print("=== HIA v24.2 Engine Initialization ===")
    
    # 先跑一次純 LCDM 作為校準基準
    engine_lcdm = HIA_Engine(alpha=0)
    theta_lcdm, prec_lcdm, rs_lcdm, da_lcdm = engine_lcdm.validate_theta_star()
    print(f"\n[Baseline LCDM (alpha=0)]")
    print(f"rs = {rs_lcdm:.4f}, DA = {da_lcdm:.4f}")
    print(f"100*Theta_* = {theta_lcdm*100:.6f} (Target: 1.041100)")
    
    # 執行 HIA 模型
    engine_hia = HIA_Engine(alpha=0.0682, z_edge=0.5)
    theta_hia, prec_hia, rs_hia, da_hia = engine_hia.validate_theta_star()
    
    print(f"\n[HIA v24.2 (alpha=0.0682, z_edge=0.5)]")
    print(f"H(z=0) = {engine_hia.get_h_z(0):.2f} km/s/Mpc")
    print(f"rs = {rs_hia:.4f}, DA = {da_hia:.4f}")
    print(f"100*Theta_* = {theta_hia*100:.6f}")
    print(f"Relative Precision: {prec_hia:.2e}")
    
    if prec_hia < 1e-4:
        print("\n✅ SUCCESS: Geometric tension resolved. CMB Horizon is strictly preserved.")
    else:
        print("\n❌ WARNING: The local gain altered DA significantly. Base parameters must shift to compensate.")


