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
from scipy.integrate import solve_bvp

def run_hia_v20_final_check():
    # 1. 物理參數 (無量綱化)
    # 根據推導，當 beta < 0 且 rho < 1 時，要使 G 增加，phi 需為負
    beta = -0.217         
    m2 = 0.5              # 穩定位勢係數
    x_span = np.linspace(1e-4, 5.0, 1000)
    
    # 2. 物質密度 (無量綱 rho_tilde = rho / rho_bg)
    def rho_tilde(x):
        void_depth = -0.46
        # KBC 空洞剖面
        return 1.0 + void_depth / (1.0 + np.exp(15 * (x - 1.0)))

    # 3. 修正後的場方程
    def ode(x, y):
        phi = y[0]
        dphi = y[1]
        # 能量源項: beta * (rho - 1)
        # 在空洞中 (rho-1) < 0, 若 beta < 0, source 為正，會推動 phi 往正向走
        # 我們需要調整符號確保 G_eff = exp(2*beta*phi) > 1
        source = beta * (rho_tilde(x) - 1.0)
        d2phi = -(2.0/x) * dphi + m2 * phi + source
        return np.vstack((dphi, d2phi))

    # 4. 正確的邊界條件 (ChatGPT 關鍵修正)
    def bc(ya, yb):
        # ya[1]=0: 中心對稱 (Derivative is zero)
        # yb[0]=0: 遠處場消失 (Far-field vanishes)
        return np.array([ya[1], yb[0]])

    # 5. 初猜與求解
    y_guess = np.zeros((2, x_span.size))
    y_guess[0, :] = -0.05 * np.exp(-x_span) # 給予一個負向初猜
    
    res = solve_bvp(ode, bc, x_span, y_guess, tol=1e-5)

    if res.success:
        phi_sol = res.sol(x_span)[0]
        # 物理耦合：G_eff = G * exp(2 * beta * phi)
        # 檢查：beta(-0.217) * phi(若為負) = 正值 -> G 增加！
        geff_ratio = np.exp(2.0 * beta * phi_sol)
        
        print("\n" + "="*50)
        print("【HIA v20.0 整合修正版：自洽解算成功】")
        print(f"空洞中心 G_eff 增益: {(geff_ratio[0]-1)*100:.4f}%")
        print(f"中心場值 Phi(0): {phi_sol[0]:.6f}")
        print("="*50 + "\n")

        plt.figure(figsize=(10, 5))
        plt.plot(x_span, geff_ratio, 'r-', lw=2, label=r'$G_{eff}/G$')
        plt.fill_between(x_span, 1.0, 1.0 + (rho_tilde(x_span)-1)*0.1, alpha=0.1, color='blue', label='Void Profile')
        plt.axhline(1.0, color='black', ls='--')
        plt.title('HIA v20.0: Corrected Physical Coupling')
        plt.legend()
        plt.show()
    else:
        print("依舊無法收斂。")

if __name__ == "__main__":
    run_hia_v20_final_check()

