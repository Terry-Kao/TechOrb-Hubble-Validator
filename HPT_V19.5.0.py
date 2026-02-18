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

def run_v19_5_fixed():
    """
    v19.5 自洽聯立求解器 (穩定版)
    修復 SyntaxWarning 並優化收斂演算法
    """
    # 1. 物理參數 (無量綱化)
    beta = -0.431          
    M_screen = 0.05        # 調整屏蔽強度以增加數值穩定性
    x_span = np.linspace(0.01, 5, 500) # 從 0.01 開始，避開中心奇點
    
    # 2. 密度分佈
    def get_rho(x):
        void_part = 1.0 - 0.46 / (1 + np.exp(15 * (x - 1)))
        galaxy_core = 20.0 * np.exp(-(x / 0.1)**2) # 稍微調低密度以利收斂
        return void_part + galaxy_core

    rho_data = get_rho(x_span)

    # 3. 聯立場方程系統
    def coupled_system(x, y):
        # y[0]=Phi, y[1]=dPhi/dx, y[2]=Psi, y[3]=dPsi/dx
        phi = y[0]
        dphi = y[1]
        psi = y[2]
        dpsi = y[3]
        
        # 數值保護：確保 phi 為正且不為零
        phi_safe = np.where(phi > 1e-7, phi, 1e-7)
        
        # (A) 資訊場方程
        v_prime = -(M_screen**4) / (phi_safe**2)
        source_phi = -beta * get_rho(x) * np.exp(2 * beta * phi_safe)
        d2phi = -(2/x) * dphi + v_prime + source_phi
        
        # (B) 修正泊松方程 (G_eff = exp(2*beta*phi))
        g_eff = np.exp(2 * beta * phi_safe)
        source_psi = g_eff * get_rho(x)
        d2psi = -(2/x) * dpsi + source_psi
        
        return np.vstack((dphi, d2phi, dpsi, d2psi))

    # 4. 邊界條件
    def bc(ya, yb):
        # ya[1]: 中心導數為 0 (對稱)
        # yb[0]: 遠處場趨於 0
        # ya[3]: 中心引力梯度為 0
        # yb[2]: 遠處引力勢設為 0
        return np.array([ya[1], yb[0], ya[3], yb[2]])

    # 5. 優化初猜 (Initial Guess)
    # 提供一個平滑下降的 phi 猜測，幫助收斂
    y_guess = np.zeros((4, x_span.size))
    y_guess[0, :] = 0.05 * np.exp(-x_span / 2) # Phi 的初猜
    y_guess[2, :] = 0.1 * np.exp(-x_span / 1)  # Psi 的初猜
    
    # 6. 執行求解
    try:
        res = solve_bvp(coupled_system, bc, x_span, y_guess, max_nodes=5000, tol=1e-3)
        
        if res.success:
            phi_sol = res.sol(x_span)[0]
            psi_sol = res.sol(x_span)[2]
            geff_ratio = np.exp(2 * beta * phi_sol)
            
            print("\n" + "="*40)
            print("【v19.5 聯立求解成功：系統已達成自洽】")
            print(f"空洞中心資訊場值 (Phi): {phi_sol[0]:.6f}")
            print(f"空洞中心 Geff/G 增益: {(geff_ratio[0]-1)*100:.4f}%")
            print(f"最大引力勢修正: {np.max(np.abs(psi_sol)):.6f}")
            print("="*40 + "\n")

            # 繪圖
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 第一子圖: 資訊場與 G 增益
            ax1.plot(x_span, phi_sol, 'b-', label=r'Information Field ($\Phi$)')
            ax1.set_ylabel(r'$\Phi$ Value', color='b')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(x_span, geff_ratio, 'r--', label=r'$G_{eff}/G$')
            ax1_twin.set_ylabel(r'Gravity Ratio', color='r')
            ax1.set_title('v19.5: Self-Consistent Field Coupling')
            ax1.legend(loc='upper right')
            
            # 第二子圖: 密度與引力勢
            ax2.fill_between(x_span, rho_data, alpha=0.2, color='green', label='Matter Density')
            ax2.set_ylabel('Density', color='g')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x_span, psi_sol, 'k-', label=r'Gravitational Potential ($\Psi$)')
            ax2_twin.set_ylabel(r'Potential $\Psi$', color='k')
            ax2.set_xlabel('Normalized Radius (x)')
            ax2.legend(loc='lower right')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("求解失敗：", res.message)
            
    except Exception as e:
        print(f"執行出錯: {e}")

if __name__ == "__main__":
    run_v19_5_fixed()

