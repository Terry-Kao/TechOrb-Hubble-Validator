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

# ==========================================
# v18.3 HIA：空間異質性 (Spatial Inhomogeneity) 引擎
# ==========================================

class SpatialActionModel:
    def __init__(self, alpha=1.2, I_crit=0.6, n_power=3.5):
        self.H0_cosmic = 67.4
        self.Om = 0.315
        self.Ol = 0.685
        
        # 核心參數：採用"晚期突變"配置
        self.alpha = alpha
        self.I_crit = I_crit
        self.n_power = n_power

    def get_I_with_environment(self, z, env_type="Average"):
        """
        [空間偵測器] 根據環境類型返回資訊密度。
        delta_I 代表局部結構相對於宇宙平均水平的偏差。
        """
        growth_factor = 1.0 / (1.0 + z)**2 # 結構隨紅移非線性增長
        
        env_map = {
            "Void": -0.8,      # 低密度區
            "Average": 0.0,    # 宇宙平均
            "Cluster": 2.5,    # 高密度結構區 (如我們所在的超星系團)
            "Super-Structure": 5.0 # 極端資訊飽和區
        }
        
        delta_I = env_map.get(env_type, 0.0)
        # 資訊密度 = 平均密度 * (1 + 局部擾動)
        return growth_factor * (1.0 + delta_I)

    def coupling_F(self, I):
        return 1.0 + self.alpha * (I**self.n_power / (I**self.n_power + self.I_crit**self.n_power))

    def calculate_H_obs(self, z_array, env_type="Average"):
        H_obs_list = []
        for z in z_array:
            E_z = np.sqrt(self.Om * (1 + z)**3 + self.Ol)
            H_bg = self.H0_cosmic * E_z
            
            # 計算 F 及其對時間的導數
            dz = 0.001
            I_now = self.get_I_with_environment(z, env_type)
            I_next = self.get_I_with_environment(z + dz, env_type)
            
            F_now = self.coupling_F(I_now)
            F_next = self.coupling_F(I_next)
            
            # dF/dt = (dF/dz) * (-H * (1+z))
            dF_dt = ((F_next - F_now) / dz) * (-H_bg * (1 + z))
            
            # 尺規修正公式
            H_obs = H_bg + 0.5 * (dF_dt / F_now)
            H_obs_list.append(H_obs)
            
        return np.array(H_obs_list)

# ==========================================
# 執行 v18.3 實戰演習：不同環境下的哈伯張力
# ==========================================
def run_v18_3_search():
    model = SpatialActionModel()
    z_vals = np.logspace(np.log10(0.01), np.log10(1.5), 100)
    
    environments = ["Void", "Average", "Cluster", "Super-Structure"]
    colors = ['gray', 'blue', 'green', 'red']
    
    plt.figure(figsize=(12, 7))
    
    print(f"{'環境類型':<15} | {'觀測 H0 (z=0.01)':<18} | {'相對於基準的偏差':<15}")
    print("-" * 55)
    
    for env, col in zip(environments, colors):
        H_obs = model.calculate_H_obs(z_vals, env_type=env)
        local_H0 = H_obs[0]
        gap = local_H0 - model.H0_cosmic
        
        print(f"{env:<15} | {local_H0:>15.2f} km/s | {gap:>12.2f}")
        
        plt.plot(z_vals, H_obs, label=f"Env: {env}", color=col, lw=2)

    # 標準 LCDM 參考線
    H_lcdm = model.H0_cosmic * np.sqrt(model.Om * (1 + z_vals)**3 + model.Ol)
    plt.plot(z_vals, H_lcdm, '--', label='Standard LCDM (Planck)', color='black', alpha=0.6)

    plt.xscale('log')
    plt.xlabel('Redshift (z)')
    plt.ylabel('H(z) [km/s/Mpc]')
    plt.title('v18.3 HIA: Local Environment Density vs. Hubble Tension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_v18_3_search()

