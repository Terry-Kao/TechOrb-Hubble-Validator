import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ---------------------------
# Physical constants & cosmology
# ---------------------------
c_km_s = 299792.458               # km / s
H0_km_s_Mpc = 67.4                # baseline H0 for background (km/s/Mpc)
Omega_m = 0.315
Omega_lambda = 1.0 - Omega_m

def H_of_z(z):
    return H0_km_s_Mpc * np.sqrt(Omega_m*(1+z)**3 + Omega_lambda)

# ---------------------------
# Field profile and derivatives
# ---------------------------
def Phi_of_z(z, Phi0):
    return Phi0 * (1.0 - np.exp(-(z / 0.5)**2))

def dPhi_dz(z, Phi0):
    return Phi0 * (2.0 * z / (0.5**2)) * np.exp(-(z / 0.5)**2)

# ---------------------------
# Disformal ansatz and photon effective speed
# ---------------------------
def c_eff(z, Phi0, D0):
    if z <= 0.0:
        return 1.0
    H = H_of_z(z)  
    dPhidz = dPhi_dz(z, Phi0)
    prefactor = (1.0 + z) * H / c_km_s  
    dPhidt = - prefactor * dPhidz      
    D_phi = D0 * Phi_of_z(z, Phi0)     
    
    arg = 1.0 - D_phi * (dPhidt**2)
    if arg <= 0.0:
        return 1e-12
    return np.sqrt(arg)

# ---------------------------
# Distances and magnitudes
# ---------------------------
def chi_mod(z, Phi0, D0):
    integrand = lambda zp: c_eff(zp, Phi0, D0) / (H_of_z(zp) / c_km_s)
    val, err = quad(integrand, 0.0, z, epsabs=1e-8, epsrel=1e-8, limit=200)
    return val  

def chi_std(z):
    integrand = lambda zp: 1.0 / (H_of_z(zp) / c_km_s)
    val, err = quad(integrand, 0.0, z, epsabs=1e-8, epsrel=1e-8, limit=200)
    return val

def D_L_mod(z, Phi0, D0):
    return (1.0 + z) * chi_mod(z, Phi0, D0)

def D_L_std(z):
    return (1.0 + z) * chi_std(z)

def mu_from_DL(DL_Mpc):
    return 5.0 * np.log10(DL_Mpc) + 25.0

# ---------------------------
# Arrival time difference EM vs GW (seconds)
# ---------------------------
Mpc_over_c_s = 3.085677581e22 / 299792458.0

def EM_GW_time_delay(z_src, Phi0, D0):
    integrand_ph = lambda zp: 1.0 / ((1.0 + zp) * (H_of_z(zp) / c_km_s) * max(c_eff(zp, Phi0, D0), 1e-16))
    integrand_gw = lambda zp: 1.0 / ((1.0 + zp) * (H_of_z(zp) / c_km_s))
    t_ph, _ = quad(integrand_ph, 0.0, z_src, epsabs=1e-8, epsrel=1e-8, limit=200)
    t_gw, _ = quad(integrand_gw, 0.0, z_src, epsabs=1e-8, epsrel=1e-8, limit=200)
    return (t_ph - t_gw) * Mpc_over_c_s

# ---------------------------
# Extended parameter search (ChatGPT's final script)
# ---------------------------
if __name__ == "__main__":
    print("=== 啟動 HIA v25.0 Disformal Ray-Tracing (Extended Search) ===")
    
    # 根據 ChatGPT 的建議，我們需要掃描極大的「負值」D0
    # 因為 D0 為負，才能讓 c_eff > 1 (光度距離被放大，超新星變暗)
    Phi0_vals = [0.2, 0.5, 1.0]
    D0_magnitudes = np.logspace(6, 14, 40)  # 1e6 to 1e14
    D0_vals = -D0_magnitudes[::-1]          # 取負值並排序
    
    z_grid = np.linspace(0.01, 0.5, 50)
    results2 = []

    for Phi0 in Phi0_vals:
        for D0 in D0_vals:
            delta_mu_vals = []
            valid = True
            for z in z_grid:
                dl_mod = D_L_mod(z, Phi0, D0)
                dl_std = D_L_std(z)
                if np.isnan(dl_mod) or dl_mod <= 0:
                    valid = False
                    break
                mu_mod = mu_from_DL(dl_mod)
                mu_std = mu_from_DL(dl_std)
                delta_mu_vals.append(mu_mod - mu_std)
                
            if not valid:
                continue
                
            avg_dm = np.mean(delta_mu_vals)
            
            try:
                dlm = D_L_mod(0.02, Phi0, D0); dls = D_L_std(0.02)
                dm_at_0p02 = mu_from_DL(dlm) - mu_from_DL(dls)
            except:
                dm_at_0p02 = np.nan
                
            # GW delay 檢測 (距離約 40 Mpc, z=0.009)
            try:
                delay_s = EM_GW_time_delay(0.009, Phi0, D0)
            except:
                delay_s = np.nan
                
            results2.append((Phi0, D0, avg_dm, dm_at_0p02, delay_s))

    results2_arr = np.array(results2, dtype=object)
    # 尋找最接近 target +0.17 mag 的參數組
    diffs2 = np.abs(np.array([r[2] for r in results2]) - 0.17)
    order2 = np.argsort(diffs2)
    top2 = results2_arr[order2][:10]

    print("\n[Top Candidates (Phi0, D0 [Mpc^2], avg Δμ, Δμ@0.02, GW delay@40Mpc [s])]")
    for row in top2:
        # 將 D0 格式化為科學記號方便閱讀
        print(f"Phi0={row[0]:.1f}, D0={row[1]:.2e}, avg_Δμ={row[2]:.4f}, Δμ@0.02={row[3]:.4f}, GW_delay={row[4]:.2e} s")

    # 繪製最佳候選者的曲線
    if len(top2) > 0:
        best2 = top2[0]
        Phi0_b, D0_b = float(best2[0]), float(best2[1])
        dm_plot2 = [mu_from_DL(D_L_mod(z, Phi0_b, D0_b)) - mu_from_DL(D_L_std(z)) for z in z_grid]

        plt.figure(figsize=(8,5))
        plt.plot(z_grid, dm_plot2, label=f'Δμ (Phi0={Phi0_b}, D0={D0_b:.2e})')
        plt.axhline(0.17, color='red', linestyle='--', label='target +0.17 mag')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Δμ (Magnitude Shift)')
        plt.legend()
        plt.grid(True)
        plt.title('HIA v25.0: Disformal Illusion Δμ(z)')
        plt.show()

        print(f"\n✅ 最佳參數組 GW170817 時間延遲: {EM_GW_time_delay(0.009, Phi0_b, D0_b):.2e} 秒")
