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
from scipy.optimize import root_scalar, minimize_scalar

class HIA_Optimizer:
    """HIA v24.5: Deep Search Physical Density Joint Optimizer"""
    def __init__(self, z_edge=0.5):
        self.z_edge = z_edge
        self.z_star = 1089.92
        self.theta_star_target = 0.010411  # Planck 2018 基準
        self.target_h0_local = 72.00       # 我們的局部觀測目標
        
        # 鎖定 Planck 2018 絕對物理密度
        h_ref = 0.674
        self.omega_m = 0.315 * h_ref**2
        self.omega_b = 0.049 * h_ref**2
        
        og_ref = 5.38e-5
        on_ref = 0.2271 * 3.046 * og_ref
        self.omega_r = (og_ref + on_ref) * h_ref**2

    def get_h_z(self, z, h0_base, alpha):
        h = h0_base / 100.0
        
        om = self.omega_m / h**2
        orad = self.omega_r / h**2
        ol = 1.0 - om - orad 
        
        # 避免非物理的負能量密度
        if ol < 0: return np.inf 
        
        E_z = np.sqrt(om * (1+z)**3 + orad * (1+z)**4 + ol)
        h_lcdm = h0_base * E_z
        gain = 1 + alpha * np.exp(-(z / self.z_edge)**2)
        return h_lcdm * gain

    def calc_theta_star(self, h0_base):
        alpha = (self.target_h0_local / h0_base) - 1.0

        def rs_integrand(z):
            h = h0_base / 100.0
            ob = self.omega_b / h**2
            og = (5.38e-5 * 0.674**2) / h**2
            R = (3.0 * ob) / (4.0 * og * (1+z))
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return cs / self.get_h_z(z, h0_base, alpha)
        
        rs, _ = quad(rs_integrand, self.z_star, 1e6)
        da, _ = quad(lambda z: 1.0 / self.get_h_z(z, h0_base, alpha), 0, self.z_star)
        
        return (rs / da)

    def objective(self, h0_base):
        return self.calc_theta_star(h0_base) - self.theta_star_target

    def run_optimization(self):
        print("🔍 啟動 HIA 深層尋優器 (區間 [50.0, 70.0])...")
        
        try:
            # 嘗試尋找跨越 0 的完美根
            res = root_scalar(self.objective, bracket=[50.0, 70.0], method='brentq')
            best_h0_base = res.root
            print("✅ 成功找到精確的零點交叉！")
        except ValueError:
            print("⚠️ 區間內未跨越零點，啟動最小化殘差模式 (尋找極限逼近解)...")
            res = minimize_scalar(lambda x: abs(self.objective(x)), bounds=(50.0, 70.0), method='bounded')
            best_h0_base = res.x
            
        best_alpha = (self.target_h0_local / best_h0_base) - 1.0
        final_theta = self.calc_theta_star(best_h0_base)
        precision = abs(final_theta - self.theta_star_target) / self.theta_star_target
        
        return best_h0_base, best_alpha, final_theta, precision

# ==========================================
# 執行
# ==========================================
if __name__ == "__main__":
    optimizer = HIA_Optimizer(z_edge=0.5)
    best_h0, best_alpha, final_theta, prec = optimizer.run_optimization()
    
    print("\n" + "="*50)
    print(" 🏆 HIA v24.5 最終物理參數鎖定報告 ")
    print("="*50)
    print(f"✅ 局部觀測 H0 目標 : {optimizer.target_h0_local:.2f} km/s/Mpc")
    print(f"🎯 真實基底 H_base  : {best_h0:.4f} km/s/Mpc")
    print(f"🎯 局部增益 Alpha   : {best_alpha:.6f}")
    print("-" * 50)
    print(f"🌌 驗證 CMB 100*Theta_* : {final_theta*100:.6f} (Target: 1.041100)")
    print(f"🌌 相對幾何誤差       : {prec:.2e}")

