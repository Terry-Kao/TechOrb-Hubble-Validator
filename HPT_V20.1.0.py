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
from scipy.integrate import solve_ivp

def run_hia_v20_shooting_method():
    """
    HIA v20.0 射擊法模擬器：繞過 BVP 收斂問題，直接積分物理場。
    """
    # 1. 物理參數 (無量綱)
    beta = -0.217         # 全息理論推導值
    m2 = 0.1              # 場的有效質量
    
    # 2. 密度剖面 (KBC Void)
    def rho_tilde(x):
        return 1.0 - 0.46 / (1 + np.exp(10 * (x - 1.0)))

    # 3. 定義場方程 (ODE)
    def ode_system(x, y):
        phi, dphi = y
        # 源項 S = beta * (rho - 1)
        source = beta * (rho_tilde(x) - 1.0)
        # 處理中心奇點 (x->0) 的極限值
        if x < 1e-6:
            d2phi = (m2 * phi + source) / 3.0  # 幾何修正
        else:
            d2phi = -(2.0/x) * dphi + m2 * phi + source
        return [dphi, d2phi]

    # 4. 執行「射擊」：假設中心場值 Phi_0 並向外演化
    # 我們嘗試幾個不同的中心值，觀察哪一個能讓場在遠處趨於 0
    phi0_candidates = [-0.1, -0.2, -0.3] 
    x_eval = np.linspace(1e-6, 5.0, 1000)
    
    plt.figure(figsize=(10, 6))
    
    for phi0 in phi0_candidates:
        sol = solve_ivp(ode_system, [1e-6, 5.0], [phi0, 0.0], t_eval=x_eval)
        
        if sol.success:
            phi_vals = sol.y[0]
            # 物理計算：G_eff = G * exp(2 * beta * phi)
            geff_ratio = np.exp(2.0 * beta * phi_vals)
            
            label = f"Initial $\Phi(0)$={phi0}"
            plt.plot(sol.t, (geff_ratio-1)*100, label=label)
            
            # 找到最接近邊界條件 (Phi_inf = 0) 的解
            if abs(phi_vals[-1]) < 0.1:
                final_phi0 = phi0
                final_gain = (geff_ratio[0]-1)*100

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axhline(10.4, color='red', lw=1, ls=':', label='Hubble Tension Target')
    plt.title('HIA v20.0: Gravitational Gain via Shooting Method')
    plt.xlabel('Distance from Void Center ($r/R_{void}$)')
    plt.ylabel('G_eff Enhancement (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"\n【模擬分析完成】")
    print(f"當 $\Phi(0)$ 設為 {final_phi0} 時，系統最穩定。")
    print(f"產生的引力增益約為: {final_gain:.2f}%")

if __name__ == "__main__":
    run_hia_v20_shooting_method()

