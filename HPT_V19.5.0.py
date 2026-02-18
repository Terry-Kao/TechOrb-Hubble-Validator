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

def run_v19_5_coupled_solver():
    """
    v19.5 自洽聯立求解器
    同時求解資訊場 Phi 與 引力勢 Psi，實現場與度規的互饋 (Feedback Loop)。
    """
    # 1. 物理參數 (無量綱化)
    beta = -0.431          # 耦合係數
    M_screen = 0.05        # 變色龍屏蔽參數
    x_span = np.linspace(0.001, 5, 1000)
    
    # 2. 物質密度分布 (KBC 空洞模型)
    def get_rho(x):
        void_part = 1.0 - 0.46 / (1 + np.exp(15 * (x - 1)))
        galaxy_core = 50.0 * np.exp(-(x / 0.05)**2)
        return void_part + galaxy_core

    rho_m = get_rho(x_span)

    # 3. 聯立場方程系統
    # y[0]=Phi, y[1]=dPhi/dx, y[2]=Psi, y[3]=dPsi/dx
    def coupled_system(x, y):
        phi = y[0]
        dphi = y[1]
        psi = y[2]
        dpsi = y[3]
        
        # 防止數值崩潰
        phi_safe = np.maximum(phi, 1e-9)
        
        # (A) 資訊場方程: Laplacian(Phi) = V'(Phi) - Beta * rho * exp(2*Beta*Phi)
        v_prime = -(M_screen**4) / (phi_safe**2)
        source_phi = -beta * get_rho(x) * np.exp(2 * beta * phi_safe)
        d2phi = -(2/x) * dphi + v_prime + source_phi
        
        # (B) 修正泊松方程: Laplacian(Psi) = 4*pi*G_eff * rho
        # 這裡簡化 4*pi*G = 1，G_eff = exp(2*Beta*Phi)
        g_eff = np.exp(2 * beta * phi_safe)
        source_psi = g_eff * get_rho(x)
        d2psi = -(2/x) * dpsi + source_psi
        
        return np.vstack((dphi, d2phi, dpsi, d2psi))

    # 4. 邊界條件
    def bc(ya, yb):
        # 中心對稱：dPhi/dx=0, dPsi/dx=0
        # 邊界穩定：Phi(R)=0, Psi(R)=特定引力勢 (此處設為 0 作為基準)
        return np.array([ya[1], yb[0], ya[3], yb[2]])

    # 5. 求解
    y_guess = np.zeros((4, x_span.size))
    y_guess[0, :] = 0.01 # Phi 初猜
    y_guess[2, :] = 0.1  # Psi 初猜
    
    res = solve_bvp(coupled_system, bc, x_span, y_guess, max_nodes=10000)

    if res.success:
        phi_sol = res.sol(x_span)[0]
        psi_sol = res.sol(x_span)[2]
        geff_ratio = np.exp(2 * beta * phi_sol)
        
        print("-" * 40)
        print("【v19.5 自洽模擬成功】")
        print(f"空洞中心 Geff 增益: {(geff_ratio[100]-1)*100:.4f}%")
        print(f"引力勢修正 Psi(0): {psi_sol[0]:.6f}")
        print("-" * 40)

        # 視覺化分析
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # 上圖：資訊場與 G 增益
        ax1.plot(x_span, phi_sol, 'b-', label='Information Field ($\Phi$)')
        ax1.set_ylabel('Field Value', color='b')
        ax1_2 = ax1.twinx()
        ax1_2.plot(x_span, geff_ratio, 'r--', label='$G_{eff}/G$')
        ax1_2.set_ylabel('Effective Gravity Ratio', color='r')
        ax1.set_title('v19.5: Coupled Information Field & Gravity')
        ax1.legend(loc='upper left')
        
        # 下圖：物質密度與引力勢
        ax2.plot(x_span, rho_m, 'g', alpha=0.3, label='Matter Density (KBC Void)')
        ax2.fill_between(x_span, rho_m, alpha=0.1, color='g')
        ax2.set_ylabel('Density', color='g')
        ax2_2 = ax2.twinx()
        ax2_2.plot(x_span, psi_sol, 'k-', label='Modified Gravitational Potential ($\Psi$)')
        ax2_2.set_ylabel('Potential Value', color='k')
        ax2.set_xlabel('Normalized Radius (x)')
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()
    else:
        print("求解失敗，系統非線性過強。")

if __name__ == "__main__":
    run_v19_5_coupled_solver()

