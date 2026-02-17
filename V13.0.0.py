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
import matplotlib.pyplot as plt
import os

# 設定隨機種子以求重現
torch.manual_seed(42)

# ==========================================
# 1. 定義 TechOrb 神經流形網絡
# ==========================================
class TechOrbManifold(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1024, output_dim=1):
        super(TechOrbManifold, self).__init__()
        
        # 1. 上帝科技球的「編碼器」：將紅移投影到高維空間
        # 我們使用多層感知機來模擬這顆球的內部複雜結構
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),  # Tanh 激活函數適合模擬平滑的物理流形
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim) # 這裡是 1024 維的 "TechOrb" 空間
        )
        
        # 2. 物理約束層 (The Poles & Equator)
        # 我們將在這裡「讀取」球體表面的狀態
        # 為了模擬球體，我們對 latent vector 做歸一化 (Normalize)
        
        # 3. 解碼器：從高維投影回我們觀測到的距離模數 mu
        # 這裡學習的是「相對於 LCDM 的偏差」 (Residual Learning)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),  # SiLU (Swish) 常用於現代物理神經網絡
            nn.Linear(64, output_dim)
        )
        
    def forward(self, z):
        # 投影到 1024 維
        latent_vector = self.encoder(z)
        
        # 強制流形約束：將向量投射到單位超球面上 (Hypersphere)
        # 這代表 "TechOrb" 的剛性結構
        norm = torch.norm(latent_vector, p=2, dim=1, keepdim=True)
        latent_sphere = latent_vector / (norm + 1e-7)
        
        # 解碼為距離修正量 (Delta mu)
        correction = self.decoder(latent_sphere)
        
        return correction, latent_sphere

# ==========================================
# 2. 物理輔助函數 (LCDM Baseline)
# ==========================================
def get_lcdm_mu(z_tensor, Om=0.3):
    # 簡單的積分近似，為了加速訓練
    # 在 PyTorch 中實現簡單的哈伯積分
    z = z_tensor.detach().numpy().flatten()
    from scipy.integrate import quad
    
    def E_inv(z_prime):
        return 1.0 / np.sqrt(Om * (1+z_prime)**3 + (1-Om))
    
    dl_list = []
    c = 299792.458
    for val in z:
        integ, _ = quad(E_inv, 0, val)
        dl_list.append((1+val) * c * integ)
    
    dl = np.array(dl_list)
    mu = 5.0 * np.log10(np.maximum(dl, 1e-5)) + 25.0
    return torch.tensor(mu, dtype=torch.float32).unsqueeze(1)

# ==========================================
# 3. 訓練與分析循環
# ==========================================
def run_v13_neural_manifold():
    print("[*] 啟動 v13.0.0：神經流形物理探測 (Neural Manifold Physics)...")
    
    # --- A. 數據加載 ---
    dat_file = "Pantheon+SH0ES.dat"
    if not os.path.exists(dat_file):
        print("[-] 數據缺失。")
        return

    df = pd.read_csv(dat_file, sep=r'\s+')
    # 為了訓練穩定，我們過濾掉 z<0.001 的極端點
    df = df[df['zHD'] > 0.001]
    
    z_train = torch.tensor(df['zHD'].values, dtype=torch.float32).unsqueeze(1)
    mu_train = torch.tensor(df['m_b_corr'].values, dtype=torch.float32).unsqueeze(1)
    
    # 計算 LCDM 基準線 (作為物理引導)
    mu_lcdm = get_lcdm_mu(z_train)
    residual_target = mu_train - mu_lcdm  # 我們讓 AI 只學習「偏差」
    
    # --- B. 模型初始化 ---
    model = TechOrbManifold(latent_dim=1024)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # --- C. 訓練循環 (尋找 TechOrb 的形狀) ---
    epochs = 2000
    print(f"[*] 開始在 1024 維空間中尋找最佳流形，共 {epochs} 輪...")
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        correction, latent_sphere = model(z_train)
        
        # 物理損失函數 (Physics-Informed Loss)
        # 1. 數據擬合誤差
        loss_data = torch.mean((correction - residual_target)**2)
        
        # 2. 流形平滑度約束 (Manifold Smoothness)
        # 我們希望這顆球是光滑的，不要有隨機噪聲，除非是必要的「褶皺」
        # 計算 latent vector 對 z 的梯度 (簡單差分近似)
        latent_grad = torch.mean(torch.abs(latent_sphere[1:] - latent_sphere[:-1]))
        loss_smooth = 0.1 * latent_grad
        
        # 3. 極軸/赤道分離約束 (可選的高級功能)
        # 這裡我們先讓網絡自由探索，看看它是否自動分離
        
        total_loss = loss_data + loss_smooth
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 200 == 0:
            print(f"    Epoch {epoch}: Loss = {total_loss.item():.6f}")

    # --- D. 結果分析與「資訊赤道」可視化 ---
    print("\n" + "="*50)
    print("   v13.0.0 神經流形診斷報告")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        pred_correction, latent_final = model(z_train)
        
        # 1. 計算最終擬合優度
        final_residual = (pred_correction - residual_target).numpy()
        rmse = np.sqrt(np.mean(final_residual**2))
        print(f" 最終擬合 RMSE: {rmse:.4f} mag")
        
        # 2. 提取「資訊複雜度」特徵 (Latent Analysis)
        # 我們對 1024 維向量做 PCA，看看主成分是否對應哈伯張力
        u, s, v = torch.pca_lowrank(latent_final, q=3)
        principal_component_1 = u[:, 0].numpy() # 假設這是主要的「資訊軸」
        
        # 3. 判斷是否有「褶皺」 (Tension Detector)
        # 檢查主要成分在低紅移 (z < 0.1) 是否有劇烈波動
        low_z_mask = (z_train.numpy().flatten() < 0.1)
        fluctuation = np.std(principal_component_1[low_z_mask])
        print(f" 低紅移區域的流形波動度: {fluctuation:.4f}")
        
        if fluctuation > 0.05:
            print(" [!] 檢測到顯著的幾何褶皺！")
            print("     這意味著在 1024 維空間中，現代宇宙處於一個不穩定的「資訊赤道」上。")
            print("     AI 發現必須扭曲時空結構才能解釋數據。")
        else:
            print(" [?] 流形相對平滑。哈伯張力可能需要更深層的物理約束。")
            
    print("="*50)
    
    # 這裡可以繪製 latent space 的圖，但在純文字介面省略
    return

if __name__ == "__main__":
    try:
        run_v13_neural_manifold()
    except Exception as e:
        print(f"執行錯誤 (可能是環境問題): {e}")


