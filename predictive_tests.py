"""
HRS Predictive Tests v1.0 - Future Observation Forecasting
---------------------------------------------------------
Purpose: Generate testable predictions for DESI (2026), Euclid, and Roman Telescope.
Focus: H(z) evolution, D_V(z) BAO scales, and Growth Tension.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- 環境檢查 ---
import subprocess, sys
def setup():
    try: import scipy, matplotlib
    except: subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "matplotlib"])
setup()

# =============================================================
# 1. 核心模型參數 (基於 v6.0 MCMC 結果)
# =============================================================
H0_HRS = 77.472
ALPHA_HRS = 0.646
H_CMB = 67.4

H0_LCDM = 70.0 # 標準模型基準 (假設值)
OM_LCDM = 0.3

# =============================================================
# 2. 物理量計算函數
# =============================================================

def h_hrs(z):
    chi = np.log(1 + z)
    return H_CMB + (H0_HRS - H_CMB) * (1.0 / np.cosh(ALPHA_HRS * chi))

def h_lcdm(z):
    return H0_LCDM * np.sqrt(OM_LCDM * (1+z)**3 + (1 - OM_LCDM))

def get_dv(z, h_func):
    """計算 BAO 觀測量 D_V(z)"""
    c = 299792.458
    def comoving_integrand(zp): return 1.0 / h_func(zp)
    dm, _ = quad(comoving_integrand, 0, z)
    dm *= c
    # D_V = [z * dm^2 / H(z)]^(1/3)
    return (z * dm**2 * c / h_func(z))**(1/3)

# =============================================================
# 3. 執行預測測試
# =============================================================

def run_predictions():
    print("="*50)
    print(" HRS 宇宙學預測報告 v1.0 - 建立可證偽性地基")
    print("="*50)
    
    # 測試紅移點 (對應 DESI 與 Euclid 重點區域)
    test_z = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    print(f"{'Redshift (z)':<15} | {'H(z) HRS':<12} | {'H(z) LCDM':<12} | {'Deviation %':<10}")
    print("-" * 55)
    
    z_plot = np.linspace(0, 3, 100)
    h_hrs_plot = [h_hrs(z) for z in z_plot]
    h_lcdm_plot = [h_lcdm(z) for z in z_plot]
    
    for z in test_z:
        h_h = h_hrs(z)
        h_l = h_lcdm(z)
        dev = (h_h - h_l) / h_l * 100
        print(f"{z:<15.2f} | {h_h:<12.3f} | {h_l:<12.3f} | {dev:<10.2f}%")

    # =============================================================
    # 4. 繪製預測圖表
    # =============================================================
    plt.figure(figsize=(10, 6))
    plt.plot(z_plot, h_hrs_plot, 'b-', label='HRS Prediction (Holographic)', linewidth=2)
    plt.plot(z_plot, h_lcdm_plot, 'r--', label='Standard LCDM (Benchmark)', linewidth=2)
    
    # 標註潛在的觀測點 (DESI/Euclid)
    plt.fill_between(z_plot, np.array(h_hrs_plot)*0.98, np.array(h_hrs_plot)*1.02, 
                     color='blue', alpha=0.1, label='HRS Uncertainty Band (2%)')
    
    plt.title("H(z) Evolution: HRS Prediction vs Standard LCDM", fontsize=14)
    plt.xlabel("Redshift (z)", fontsize=12)
    plt.ylabel("Expansion Rate H(z) [km/s/Mpc]", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存結果
    plt.savefig("hrs_predictive_h_z.png")
    print("\n[🎉] 預測圖表已儲存：'hrs_predictive_h_z.png'")
    
    # --- 關鍵宣告 ---
    print("\n[📢] 核心物理宣告：")
    print(f"1. 在 z=1.5 處，HRS 預測 H(z) 應為 {h_hrs(1.5):.2f}。")
    print("2. 如果未來觀測數據在此紅移處低於此值 5% 以上，則 HRS 投影假說需修正。")
    print("3. 這是一個強大的『預先聲明』，用於對抗後置擬合的質疑。")

if __name__ == "__main__":
    run_predictions()
