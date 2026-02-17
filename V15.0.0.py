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

# ==========================================
# 1. 定義「斷層跳變」模型 (Fault-Line Model)
# ==========================================
class FaultLineModel(nn.Module):
    def __init__(self):
        super(FaultLineModel, self).__init__()
        
        # 參數 1: 斷層發生的位置 (Critical Redshift)
        self.zc = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        
        # 參數 2: 跳變的幅度 (Jump Amplitude)
        # 代表跨越斷層後的「折射率」或 $H$ 的修正比
        self.delta_h = nn.Parameter(torch.tensor([0.08], dtype=torch.float32))
        
        # 參數 3: 斷層的寬度 (Sharpness)
        # 我們設定一個非常大的值，模擬「瞬間」跳變
        self.sharpness = 50.0 

    def forward(self, z):
        # 使用 Sigmoid 模擬 Heaviside 階梯函數
        # 當 z < zc 時，switch -> 1 (近場)
        # 當 z > zc 時，switch -> 0 (深空)
        switch = torch.sigmoid(self.sharpness * (self.zc - z))
        
        # 構建折射因子：近場時 H 增加 (1 + delta_h)，深空時不變 (1.0)
        refraction_factor = 1.0 + self.delta_h * switch
        
        return refraction_factor

# ==========================================
# 2. 物理演算核心
# ==========================================
def get_lcdm_mu(z_tensor, Om=0.3):
    z = z_tensor.detach().numpy().flatten()
    from scipy.integrate import quad
    def E_inv(z_p): return 1.0 / np.sqrt(Om * (1+z_p)**3 + (1-Om))
    c = 299792.458
    dl = np.array([(1+val) * c * quad(E_inv, 0, val)[0] for val in z])
    return torch.tensor(5.0 * np.log10(np.maximum(dl, 1e-5)) + 25.0, dtype=torch.float32).unsqueeze(1)

# ==========================================
# 3. 掃描執行
# ==========================================
def run_v15_fault_scan():
    print("[*] 啟動 v15.0.0：折射斷層掃描 (RFS)...")
    
    # 加載數據
    dat_file = "Pantheon+SH0ES.dat"
    if not os.path.exists(dat_file): return print("[-] 數據缺失")
    df = pd.read_csv(dat_file, sep=r'\s+')
    
    z_train = torch.tensor(df['zHD'].values, dtype=torch.float32).unsqueeze(1)
    mu_obs = torch.tensor(df['m_b_corr'].values, dtype=torch.float32).unsqueeze(1)
    mu_lcdm = get_lcdm_mu(z_train)
    
    # 初始化模型
    model = FaultLineModel()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("[*] 正在測繪斷層位置與跳變強度...")
    for epoch in range(1001):
        optimizer.zero_grad()
        
        # 1. 獲取當前紅移處的「折射因子」
        refraction = model(z_train)
        
        # 2. 計算修正後的距離模數
        # 邏輯：H_obs = H_lcdm * refraction
        # 則 mu_obs = mu_lcdm - 5 * log10(refraction)
        mu_pred = mu_lcdm - 5.0 * torch.log10(refraction)
        
        loss = torch.mean((mu_pred - mu_obs)**2)
        loss.backward()
        optimizer.step()
        
        if epoch % 250 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.6f} | zc = {model.zc.item():.4f}")

    # --- 最終診斷 ---
    print("\n" + "="*50)
    print("   v15.0.0 斷層掃描診斷報告")
    print("="*50)
    zc_final = model.zc.item()
    dh_final = model.delta_h.item()
    rmse = np.sqrt(loss.item())
    
    print(f" 檢測到斷層紅移 (z_c): {zc_final:.4f}")
    print(f" 跨維度跳變幅度 (Delta H): {dh_final*100:.2f}%")
    print(f" 最終擬合 RMSE: {rmse:.4f} mag")
    print("-" * 50)
    
    # 判讀
    if 0.01 < zc_final < 0.1:
        print(f" [!] 警告：在極近場 (z={zc_final:.3f}) 發現結構性斷裂。")
        print("     這代表我們所在的局部區域與整個宇宙的物理規則存在「相位差」。")
    elif zc_final >= 0.1:
        print(f" [!] 發現演化斷層。宇宙在 z={zc_final:.3f} 時經歷了架構升級。")
    
    if rmse < 0.15:
        print(" [***] 突破！該斷層模型完美擬合了哈伯張力數據。")
    else:
        print(" [?] 斷層模型雖有改進，但可能需要多個斷層同時存在。")
    print("="*50)

if __name__ == "__main__":
    run_v15_fault_scan()

