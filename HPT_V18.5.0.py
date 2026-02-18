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

class HIASectorStabilized:
    def __init__(self):
        self.H0_cosmic = 67.4
        self.Om = 0.315
        self.Ol = 0.685
        self.real_world_delta = -0.46 # KBC 空洞真實觀測值

    def coupling_F_stabilized(self, I, alpha, beta):
        """
        [穩定化耦合函數] 使用 Tanh 替代 Power Law
        alpha: 修正的最大強度上限
        beta: 響應靈敏度 (取代 n_power)
        """
        # F(I) = 1 + alpha * tanh(beta * I)
        return 1.0 + alpha * np.tanh(beta * I)

    def predict_hubble(self, z, delta, alpha, beta):
        dz = 0.001
        E_z = np.sqrt(self.Om * (1 + z)**3 + self.Ol)
        H_bg = self.H0_cosmic * E_z
        
        # 結構增長 + 局部密度
        growth = 1.0 / (1.0 + z)**2
        I_now = growth * (1.0 + delta)
        I_next = (1.0 / (1.0 + z + dz)**2) * (1.0 + delta)
        
        F_now = self.coupling_F_stabilized(I_now, alpha, beta)
        F_next = self.coupling_F_stabilized(I_next, alpha, beta)
        
        # H_obs = H_bg + 0.5 * (dF/dt / F)
        dF_dt = ((F_next - F_now) / dz) * (-H_bg * (1 + z))
        return H_bg + 0.5 * (dF_dt / F_now)

    def run_reverse_search(self):
        print("--- [v18.5 HIA 逆向擬合與穩定性測試] ---")
        target_H0 = 73.04
        best_alpha, best_beta = 0, 0
        min_residual = float('inf')
        
        # 廣域搜索最穩定的物理常數
        for a in np.linspace(0.1, 1.0, 50):
            for b in np.linspace(0.1, 2.0, 50):
                h0_pred = self.predict_hubble(0.01, self.real_world_delta, a, b)
                residual = abs(h0_pred - target_H0)
                
                if residual < min_residual:
                    min_residual = residual
                    best_alpha, best_beta = a, b
        
        print(f"找到最優穩定參數:")
        print(f" > Alpha (最大修正力): {best_alpha:.4f}")
        print(f" > Beta (響應靈敏度): {best_beta:.4f}")
        print(f" > 預測 H0: {self.predict_hubble(0.01, self.real_world_delta, best_alpha, best_beta):.4f}")
        print(f" > 殘差: {min_residual:.4f}")
        print("-" * 40)
        
        return best_alpha, best_beta

# 執行搜索
engine = HIASectorStabilized()
alpha_fix, beta_fix = engine.run_reverse_search()

# 驗證在不同環境下的表現 (使用新穩定參數)
envs = {"Real KBC Void": -0.46, "Average": 0.0, "High Cluster": 1.5}
print("\n[v18.5 穩定型環境壓力測試]")
for name, d in envs.items():
    h_res = engine.predict_hubble(0.01, d, alpha_fix, beta_fix)
    print(f"{name:<15}: {h_res:.2f} km/s/Mpc")


