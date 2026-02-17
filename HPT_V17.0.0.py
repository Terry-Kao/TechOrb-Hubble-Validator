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
from scipy.linalg import expm

# 設定隨機種子以求結果可重現
np.random.seed(2026)

class HolographicToyModel:
    def __init__(self):
        # --- 1. 定義 6 維「上帝科技球」的本體結構 ---
        # 前 4 維是標準時空 (t, x, y, z)，刻度為 1.0
        # 後 2 維是額外維度 (Extra 1, Extra 2)，刻度設定略大於 1
        # 這代表如果投影偏轉到這些維度，我們會測量到「更長的距離」或「更快的膨脹」
        self.D_dims = 6
        self.metric_diagonal = np.array([1.0, 1.0, 1.0, 1.0, 1.15, 1.20])
        
        # 旋轉靈敏度 (Sensitivity): 資訊熵對角度的影響係數
        # 我們希望這是一個微擾，而不是劇烈翻轉
        self.rotation_k = 0.15  # 弧度係數

    def get_entropy(self, z):
        """
        模擬資訊熵密度 S(z)。
        假設：隨著宇宙演化 (z 變小)，結構形成，資訊熵 S 增加。
        S(z=0) = 1.0 (現代)
        S(z=1100) -> 0.0 (早期)
        """
        return 1.0 / (1.0 + z)

    def get_projection_matrix(self, S):
        """
        核心函數：計算隨熵 S 變化的投影算子 P(S)。
        這裡模擬一個 6D -> 4D 的旋轉投影。
        """
        # 1. 計算旋轉角度 theta (隨熵增加而變大)
        theta = self.rotation_k * S
        
        # 2. 構建旋轉矩陣 R (在 4D 和 5D 之間發生混合)
        # 這模擬了我們的觀測視角從標準維度稍微「滑」向了額外維度
        # 使用生成元構建旋轉 (Lie Algebra approach for smoothness)
        generator = np.zeros((self.D_dims, self.D_dims))
        
        # 讓 x 軸 (dim 1) 與 Extra 1 (dim 4) 發生耦合
        generator[1, 4] = theta
        generator[4, 1] = -theta
        
        # 讓 y 軸 (dim 2) 與 Extra 2 (dim 5) 發生耦合
        generator[2, 5] = theta * 0.5
        generator[5, 2] = -theta * 0.5
        
        R = expm(generator)
        
        # 3. 投影切片：我們只取前 4 個維度作為觀測結果
        # 這就是我們的「視網膜」
        Projection = R[:4, :] 
        
        return Projection

    def observe_expansion_rate(self, z_array):
        """
        計算觀測到的有效哈伯常數 H_eff
        """
        H_ref = 67.4  # Planck 2018 基準值 (LCDM)
        H_obs_list = []
        distortion_list = []
        angle_list = []

        for z in z_array:
            S = self.get_entropy(z)
            P = self.get_projection_matrix(S)
            
            # --- 投影幾何計算 ---
            # 有效度規 g_eff = P * G_6D * P_transpose
            # 我們計算投影後的「體積縮放因子」或「跡(Trace)的變化」
            
            # 原始 4D 度規的跡 (應該是 4.0)
            trace_original = 4.0 
            
            # 投影後的有效度規
            G_6D = np.diag(self.metric_diagonal)
            g_eff = P @ G_6D @ P.T
            
            # 計算跡的膨脹比率 (Trace scaling)
            # 這代表了單位長度在投影后變長了多少
            trace_projected = np.trace(g_eff)
            
            # 視差因子 (Parallax Factor)
            # 幾何上的長度放大會導致測量到的 H 變大
            distortion = np.sqrt(trace_projected / trace_original)
            
            H_obs = H_ref * distortion
            
            H_obs_list.append(H_obs)
            distortion_list.append(distortion)
            angle_list.append(self.rotation_k * S * (180/np.pi)) # 轉成角度記錄

        return np.array(H_obs_list), np.array(distortion_list), np.array(angle_list)

# ==========================================
# 執行 v17.0.0 模擬
# ==========================================
def run_v17_simulation():
    print("[*] 啟動 v17.0.0：全息投影視差 (HPT) 玩具模型模擬...")
    
    model = HolographicToyModel()
    
    # 生成紅移序列 (從 z=0 到 z=1100，對數分佈)
    z_vals = np.concatenate([
        np.linspace(0, 2, 100),       # 近場 (SNIa)
        np.linspace(2, 1100, 100)     # 深空 (CMB)
    ])
    z_vals = np.sort(np.unique(z_vals))
    
    H_obs, distortions, angles = model.observe_expansion_rate(z_vals)
    
    # --- 結果診斷 ---
    H0_local = H_obs[0]     # z=0
    H0_deep = H_obs[-1]     # z=1100
    Max_Angle = angles[0]   # z=0 時的最大旋轉角
    
    print(f"\n[模擬結果分析]")
    print(f"基準 H0 (Planck): 67.4 km/s/Mpc")
    print(f"-"*40)
    print(f"模擬 H0 (Local, z=0):    {H0_local:.2f} km/s/Mpc")
    print(f"模擬 H0 (Deep, z=1100):  {H0_deep:.2f} km/s/Mpc")
    print(f"-"*40)
    print(f"哈伯張力 (Delta H):      {H0_local - H0_deep:.2f} km/s/Mpc")
    print(f"相對偏差 (Percentage):   {((H0_local - H0_deep)/H0_deep)*100:.2f}%")
    print(f"最大投影旋轉角:          {Max_Angle:.2f} 度")
    
    # --- 繪圖驗證 ---
    plt.figure(figsize=(12, 5))
    
    # 左圖：哈伯常數隨紅移的視差漂移
    plt.subplot(1, 2, 1)
    plt.plot(z_vals, H_obs, label='Holographic H(z)', color='blue', linewidth=2)
    plt.axhline(y=67.4, color='green', linestyle='--', label='Planck Baseline (67.4)')
    plt.axhline(y=73.0, color='red', linestyle='--', label='SH0ES Local (73.0)')
    plt.xscale('log')
    plt.xlabel('Redshift z (Log Scale)')
    plt.ylabel('Observed H0 [km/s/Mpc]')
    plt.title('HPT: Projection Parallax Effect')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 右圖：旋轉角度與失真因子
    plt.subplot(1, 2, 2)
    plt.plot(z_vals, distortions, label='Distortion Factor', color='purple')
    plt.xscale('log')
    plt.xlabel('Redshift z (Log Scale)')
    plt.ylabel('Geometric Scaling (Reference=1.0)')
    plt.title('Manifold Projection Distortion')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.show()

    # --- 關鍵判定 ---
    print("\n[v17.0.0 判定標準]")
    if 72.0 <= H0_local <= 74.0 and abs(H0_deep - 67.4) < 0.5:
        print("[SUCCESS] 模擬成功！")
        print(">> 模型在保持深空 CMB 錨點不變的同時，僅通過幾何視差重現了近場哈伯張力。")
        print(f">> 關鍵發現：只需旋轉 {Max_Angle:.2f} 度，就能產生 9% 的觀測差異。")
        print(">> 這證明了「哈伯張力是投影視差」在數學上是完全可行的。")
    else:
        print("[FAIL] 模擬未達標。")
        print(">> 參數設置未能重現觀測特徵，或旋轉角度過大導致物理崩潰。")

if __name__ == "__main__":
    run_v17_simulation()



