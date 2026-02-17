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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# 設定隨機種子
torch.manual_seed(2026)
np.random.seed(2026)

# ==========================================
# 1. 定義「跨維度耦合」網絡 (CDC Network)
# ==========================================
class CouplingManifold(nn.Module):
    def __init__(self):
        super(CouplingManifold, self).__init__()
        
        # 輸入維度 = 2 (時間 z, 能量密度 rho)
        # 隱藏層寬度增加，以捕捉複雜的交互作用
        self.coupling_surface = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),          # SiLU (Swish) 平滑且具備非線性
            nn.Linear(64, 128),
            nn.Tanh(),          # Tanh 用於模擬物理場的有界波動
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)    # 輸出：耦合因子 G (Coupling Factor)
        )
        
    def forward(self, z, rho):
        # 將兩個維度合併為輸入向量
        inputs = torch.cat([z, rho], dim=1)
        
        # 預測耦合因子的 "擾動量" (Delta)
        # 初始狀態接近 0，代表標準物理
        delta_G = self.coupling_surface(inputs)
        
        # 最終耦合因子 G = 1.0 + delta
        # 我們限制擾動在 +/- 20% 以內，避免物理崩潰
        G = 1.0 + 0.2 * torch.tanh(delta_G)
        
        return G

# ==========================================
# 2. 物理計算核心
# ==========================================
def get_energy_density(z_tensor, Om=0.3):
    # 計算無量綱的能量密度背景
    # rho ~ Om*(1+z)^3 + Ol
    rho = Om * (1 + z_tensor)**3 + (1 - Om)
    # 取對數以標準化輸入範圍 (Log-space is better for neural nets)
    return torch.log10(rho)

def get_lcdm_mu(z_tensor, Om=0.3):
    # 標準 LCDM 的距離模數 (基準線)
    z = z_tensor.detach().numpy().flatten()
    from scipy.integrate import quad
    
    def E_inv(z_prime):
        return 1.0 / np.sqrt(Om * (1+z_prime)**3 + (1-Om))
    
    c = 299792.458
    dl_list = [(1+val) * c * quad(E_inv, 0, val)[0] for val in z]
    dl = np.array(dl_list)
    return torch.tensor(5.0 * np.log10(np.maximum(dl, 1e-5)) + 25.0, dtype=torch.float32).unsqueeze(1)

# ==========================================
# 3. 訓練與診斷循環
# ==========================================
def run_v14_coupling():
    print("[*] 啟動 v14.0.0：跨維度耦合 (CDC) 探測...")
    
    # --- A. 數據準備 ---
    dat_file = "Pantheon+SH0ES.dat"
    if not os.path.exists(dat_file):
        print("[-] 數據文件缺失。")
        return

    df = pd.read_csv(dat_file, sep=r'\s+')
    df = df[df['zHD'] > 0.005] # 過濾極近場雜訊
    
    z_train = torch.tensor(df['zHD'].values, dtype=torch.float32).unsqueeze(1)
    mu_obs = torch.tensor(df['m_b_corr'].values, dtype=torch.float32).unsqueeze(1)
    
    # 計算第二維度：能量密度
    rho_train = get_energy_density(z_train)
    
    # 計算基準物理 (LCDM)
    mu_lcdm = get_lcdm_mu(z_train)
    
    # --- B. 初始化模型 ---
    model = CouplingManifold()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    # --- C. 訓練：尋找耦合面 ---
    print(f"[*] 開始掃描「時間-能量」耦合面...")
    epochs = 1500
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 1. 獲取耦合因子 G(z, rho)
        G_factor = model(z_train, rho_train)
        
        # 2. 物理預測：修正後的距離模數
        # 如果 G 改變了引力強度或光子能量，距離模數會發生偏移
        # mu_new = mu_lcdm - 5 * log10(G) (假設 G 影響光度距離的有效縮放)
        mu_pred = mu_lcdm - 5.0 * torch.log10(G_factor)
        
        # 3. 損失函數
        loss_fit = torch.mean((mu_pred - mu_obs)**2)
        
        # 正則化：我們希望耦合面是平滑的，不要過度扭曲
        loss_smooth = 0.05 * torch.mean(torch.abs(G_factor[1:] - G_factor[:-1]))
        
        total_loss = loss_fit + loss_smooth
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 300 == 0:
            print(f"    Epoch {epoch}: Loss = {total_loss.item():.6f}")

    # --- D. 最終診斷：這個面扭曲了嗎？ ---
    print("\n" + "="*50)
    print("   v14.0.0 跨維度耦合診斷")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        G_final = model(z_train, rho_train).numpy().flatten()
        z_vals = z_train.numpy().flatten()
        
        # 統計 G 的特徵
        mean_G = np.mean(G_final)
        max_deviation = np.max(np.abs(G_final - 1.0))
        
        # 尋找扭曲最大的位置 (The Twist Point)
        max_idx = np.argmax(np.abs(G_final - 1.0))
        z_at_twist = z_vals[max_idx]
        
        print(f" 平均耦合因子 <G>: {mean_G:.5f}")
        print(f" 最大耦合偏差 (Max Twist): {max_deviation:.5f}")
        print(f" 扭曲發生紅移 (z_twist): {z_at_twist:.4f}")
        
        # 判讀邏輯
        rmse = np.sqrt(np.mean((mu_pred.numpy() - mu_obs.numpy())**2))
        print(f" 最終擬合 RMSE: {rmse:.4f} mag")
        print("-" * 50)
        
        if max_deviation > 0.02:
            print(" [!] 發現顯著的「耦合扭曲」！")
            print(f"     在 z = {z_at_twist:.3f} 處，能量與時間的交互作用導致物理常數偏離了 {max_deviation*100:.2f}%。")
            if z_at_twist < 0.1:
                print("     >> 這是「近場效應」：現代宇宙的真空能主導導致了架構變形。")
            else:
                print("     >> 這是「演化效應」：宇宙中期的某個事件改變了常數。")
        else:
            print(" [?] 耦合面依然平坦。")
            print("     即使引入能量軸，AI 仍認為常數是恆定的。")
            
    print("="*50)

if __name__ == "__main__":
    try:
        run_v14_coupling()
    except Exception as e:
        print(f"執行錯誤: {e}")

