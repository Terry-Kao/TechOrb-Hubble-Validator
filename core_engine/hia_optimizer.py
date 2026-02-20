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
from scipy.optimize import root_scalar

class HIA_Optimizer:
    """HIA v24.4: Physical Density Locked Joint Optimization"""
    def __init__(self, z_edge=0.5):
        self.z_edge = z_edge
        self.z_star = 1089.92
        self.theta_star_target = 0.010411  # Planck 2018 基準
        self.target_h0_local = 72.00       # 我們的局部觀測目標
        
        # 1. 鎖定 Planck 2018 的「絕對物理密度 (omega = Omega * h^2)」
        # 這是保證早期宇宙 (z > 1000) 絕對不變的唯一法則
        h_ref = 0.674
        self.omega_m = 0.315 * h_ref**2     # 物理物質密度
        self.omega_b = 0.049 * h_ref**2     # 物理重子密度
        
        og_ref = 5.38e-5
        on_ref = 0.2271 * 3.046 * og_ref
        self.omega_r = (og_ref + on_ref) * h_ref**2 # 物理輻射密度

    def get_h_z(self, z, h0_base, alpha):
        h = h0_base / 100.0
        
        # 2. 根據新的 h0_base，動態反推相對百分比 (Omega)
        om = self.omega_m / h**2
        orad = self.omega_r / h**2
        ol = 1.0 - om - orad # 暗能量佔比被動調整
        
        # 計算背景 H_LCDM
        E_z = np.sqrt(om * (1+z)**3 + orad * (1+z)**4 + ol)
        h_lcdm = h0_base * E_z
        
        # 加上 HIA 局部增益
        gain = 1 + alpha * np.exp(-(z / self.z_edge)**2)
        return h_lcdm * gain

    def calc_theta_star(self, h0_base):
        # 確保 H(0) = 72.00，動態連動 alpha
        alpha = (self.target_h0_local / h0_base) - 1.0

        def rs_integrand(z):
            h = h0_base / 100.0
            ob = self.omega_b / h**2
            og = (5.38e-5 * 0.674**2) / h**2 # 僅光子
            
            R = (3.0 * ob) / (4.0 * og * (1+z))
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return cs / self.get_h_z(z, h0_base, alpha)
        
        rs, _ = quad(rs_integrand, self.z_star, 1e6)
        da, _ = quad(lambda z: 1.0 / self.get_h_z(z, h0_base, alpha), 0, self.z_star)
        
        return (rs / da)

    def objective(self, h0_base):
        return self.calc_theta_star(h0_base) - self.theta_star_target

    def run_optimization(self):
        print("🔍 啟動 HIA 聯合補償尋優器 (物理密度鎖定模式)...")
        
        # 預先診斷邊界
        val_62 = self.objective(62.0)
        val_68 = self.objective(68.0)
        print(f"   [診斷] f(62.0) = {val_62:e} (若為負代表 theta_star 過小)")
        print(f"   [診斷] f(68.0) = {val_68:e} (若為正代表 theta_star 過大)")
        
        # 使用 brentq 尋找跨越 0 的完美根
        res = root_scalar(self.objective, bracket=[62.0, 68.0], method='brentq')
        
        if res.converged:
            best_h0_base = res.root
            best_alpha = (self.target_h0_local / best_h0_base) - 1.0
            
            # 重新計算驗證
            final_theta = self.calc_theta_star(best_h0_base)
            precision = abs(final_theta - self.theta_star_target) / self.theta_star_target
            
            return best_h0_base, best_alpha, final_theta, precision
        else:
            return None, None, None, None

# ==========================================
# 執行尋優
# ==========================================
if __name__ == "__main__":
    optimizer = HIA_Optimizer(z_edge=0.5)
    best_h0, best_alpha, final_theta, prec = optimizer.run_optimization()
    
    if best_h0:
        print("\n" + "="*50)
        print(" 🏆 HIA v24.4 最終黃金參數鎖定 (Physical Joint Fit) ")
        print("="*50)
        print(f"✅ 局部 Hubble 目標 : {optimizer.target_h0_local:.2f} km/s/Mpc")
        print(f"✅ 尋優得出 H_base  : {best_h0:.4f} km/s/Mpc (真實背景膨脹率)")
        print(f"✅ 尋優得出 Alpha   : {best_alpha:.6f} (全息局部增益)")
        print("-" * 50)
        print(f"🎯 驗證 CMB 100*Theta_* : {final_theta*100:.6f} (Target: 1.041100)")
        print(f"🎯 最終相對幾何誤差       : {prec:.2e}")
        print("🔥 結論：數學與物理的完美閉環！")

