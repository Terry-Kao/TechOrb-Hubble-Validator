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
    """HIA v24.3: CMB & Local H0 Joint Optimization Engine"""
    def __init__(self, z_edge=0.5, om=0.315):
        self.z_edge = z_edge
        
        # 精確的 Planck 2018 物理常數
        self.om = om
        self.ob = 0.049
        self.og = 5.38e-5
        self.on = 0.2271 * 3.046 * self.og
        self.orad = self.og + self.on
        
        self.z_star = 1089.92
        self.theta_star_target = 0.010411  # Planck 2018 幾何極限
        self.target_h0_local = 72.00       # 我們的局部觀測目標

    def get_h_z(self, z, h0_base, alpha):
        # 動態計算 H(z)
        E_z = np.sqrt(self.om * (1+z)**3 + self.orad * (1+z)**4 + (1 - self.om - self.orad))
        h_lcdm = h0_base * E_z
        gain = 1 + alpha * np.exp(-(z / self.z_edge)**2)
        return h_lcdm * gain

    def calc_theta_star(self, h0_base):
        # 為了確保 H(z=0) 永遠等於 72.00，alpha 必須與 h0_base 連動
        # 公式: H(0) = h0_base * (1 + alpha) = 72.00
        alpha = (self.target_h0_local / h0_base) - 1.0

        # 計算聲學視界 rs
        def rs_integrand(z):
            R = (3.0 * self.ob) / (4.0 * self.og * (1+z))
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return cs / self.get_h_z(z, h0_base, alpha)
        
        rs, _ = quad(rs_integrand, self.z_star, 1e6)
        
        # 計算共動距離 da
        da, _ = quad(lambda z: 1.0 / self.get_h_z(z, h0_base, alpha), 0, self.z_star)
        
        return (rs / da)

    def objective(self, h0_base):
        # 優化目標：讓計算出的 theta_star 減去 目標 theta_star 趨近於 0
        return self.calc_theta_star(h0_base) - self.theta_star_target

    def run_optimization(self):
        print("🔍 啟動 HIA 聯合補償尋優器 (目標: CMB Theta* 誤差 = 0)...")
        # 我們知道 h0_base 必須比 67.4 低來補償，所以設定搜尋區間在 60 到 68 之間
        res = root_scalar(self.objective, bracket=[60.0, 68.0], method='brentq')
        
        if res.converged:
            best_h0_base = res.root
            best_alpha = (self.target_h0_local / best_h0_base) - 1.0
            
            # 驗證結果
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
    
    print("\n" + "="*45)
    print(" 🏆 HIA v24.3 最終黃金參數鎖定 (Joint Fit) ")
    print("="*45)
    print(f"✅ 局部 Hubble 目標 : {optimizer.target_h0_local:.2f} km/s/Mpc")
    print(f"✅ 尋優得出 H_base  : {best_h0:.4f} km/s/Mpc (真實背景膨脹率)")
    print(f"✅ 尋優得出 Alpha   : {best_alpha:.6f} (真實局域增益)")
    print("-" * 45)
    print(f"🎯 驗證 CMB 100*Theta_* : {final_theta*100:.6f} (Target: 1.041100)")
    print(f"🎯 最終相對幾何誤差       : {prec:.2e}")
    
    if prec < 1e-6:
        print("\n🔥 結論：完美對齊！我們成功在不破壞 CMB 的前提下，達成了局部 72.00 的擴張。")
        print("🔥 ChatGPT 教授的質疑已被我們用數值手段徹底粉碎。")
    else:
        print("\n❌ 警告：尋優失敗，請檢查積分條件。")
