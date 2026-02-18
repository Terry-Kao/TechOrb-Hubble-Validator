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

def run_v19_4_final():
    """
    v19.4 HIA 資訊場模擬：整合變色龍屏蔽機制
    目標：驗證資訊場在星系內部(高密度)被抑制，在空洞(低密度)激發。
    """
    # 1. 物理參數設定 (無量綱化)
    # x=1 對應 350 Mpc (KBC空洞半徑)
    beta = -0.431          # 資訊-物質耦合係數
    M_screen = 0.05        # 屏蔽閾值參數 (控制屏蔽啟動的靈敏度)
    delta_void = -0.46     # KBC 空洞密度虧損
    
    # 定義密度分佈函數 rho(x)
    def get_rho(x):
        # 局部高密度核心 (模擬星系/地球，密度是背景的 100 倍)
        galaxy_core = 100.0 * np.exp(-(x / 0.02)**2)
        # 大尺度空洞分佈
        void_profile = 1.0 + delta_void / (1 + np.exp(15 * (x - 1)))
        return galaxy_core + void_part if 'void_part' in locals() else galaxy_core + void_profile

    # 2. 定義場方程系統 (ODE)
    # Laplacian(phi) = V'(phi) - beta * (rho - rho_bg)
    # y[0] = phi, y[1] = dphi/dx
    def ode_system(x, y):
        phi, dphi = y
        # 變色龍位勢導數 V'(phi) = -M^4 / phi^2 (Ratra-Peebles 型)
        # 加入 1e-9 防止 phi 為 0 時計算崩潰
        v_prime = - (M_screen**4) / (phi**2 + 1e-9)
        
        # 物質源項 (只有偏離背景密度 rho=1 的部分會激發場)
        rho = get_rho(x)
        source = - beta * (rho - 1.0)
        
        # 球對稱下的 Laplacian: phi'' + (2/x)phi'
        d2phi = -(2/x) * dphi + v_prime + source
        return np.vstack((dphi, d2phi))

    # 3. 邊界條件 (Boundary Conditions)
    def bc(ya, yb):
        # ya[1]: 中心導數為 0 (物理對稱性)
        # yb[0]: 遠處場值趨近於某個極小穩定值 (這裡設為 0)
        return np.array([ya[1], yb[0]])

    # 4. 數值求解
    x_span = np.linspace(0.001, 5, 1000) # 從 0.001 開始避免 2/x 奇點
    y_init = np.ones((2, x_span.size)) * 0.1 # 初猜場值
    
    res = solve_bvp(ode_system, bc, x_span, y_init, max_nodes=5000)

    # 5. 結果處理與視覺化
    if res.success:
        phi = res.sol(x_span)[0]
        rho = get_rho(x_span)
        # 有效引力偏離 Geff/G = exp(2 * beta * phi)
        geff_ratio = np.exp(2 * beta * phi)
        
        # 輸出關鍵點數據
        idx_galaxy = 0  # 近中心(星系)
        idx_void = 100  # 空洞中心區域 (約 x=0.5)
        
        print("-" * 30)
        print(f"【v19.4 模擬結果】")
        print(f"星系中心 (x~0.01) Geff/G: {geff_ratio[idx_galaxy]:.8f} (應接近 1.0)")
        print(f"空洞內部 (x~0.50) Geff/G: {geff_ratio[idx_void]:.8f} (應 > 1.0)")
        print(f"生成的有效 Alpha: {geff_ratio[idx_void] - 1:.6f}")
        print("-" * 30)

        # 繪圖
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Normalized Radius (x)')
        ax1.set_ylabel('Field Value (phi) / Geff Ratio', color='tab:blue')
        ax1.plot(x_span, phi, label='Information Field (phi)', color='tab:blue', lw=2)
        ax1.plot(x_span, geff_ratio, '--', label='G_eff / G', color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylim(0.95, 1.2)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Density (rho)', color='tab:red')
        ax2.plot(x_span, rho, label='Matter Density (rho)', color='tab:red', alpha=0.3)
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('HIA v19.4: Chameleon Screening in KBC Void')
        fig.tight_layout()
        plt.legend(loc='upper right')
        plt.show()
    else:
        print("求解失敗，請檢查屏蔽參數 M_screen 是否過大導致剛性問題。")

if __name__ == "__main__":
    run_v19_4_final()

