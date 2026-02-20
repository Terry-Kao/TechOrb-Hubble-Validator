import subprocess
import sys

# --- 自動環境檢查機制 ---
def setup_environment():
    required = {"numpy", "pandas", "matplotlib", "scipy", "requests", "emcee", "corner"}
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
from scipy.integrate import quad
from scipy.optimize import root_scalar, minimize_scalar

class HIA_Optimizer:
    """HIA v24.5: Deep Search Physical Density Joint Optimizer"""
    def __init__(self, z_edge=0.5):
        self.z_edge = z_edge
        self.z_star = 1089.92
        self.theta_star_target = 0.010411  # Planck 2018 limit
        self.target_h0_local = 72.00       # Pantheon+ / SH0ES target
        
        # Absolute Physical Density Locking (omega = Omega * h^2)
        h_ref = 0.674
        self.omega_m = 0.315 * h_ref**2
        self.omega_b = 0.049 * h_ref**2
        
        og_ref = 5.38e-5
        on_ref = 0.2271 * 3.046 * og_ref
        self.omega_r = (og_ref + on_ref) * h_ref**2

    def get_h_z(self, z, h0_base, alpha):
        h = h0_base / 100.0
        om = self.omega_m / h**2
        orad = self.omega_r / h**2
        ol = 1.0 - om - orad 
        
        if ol < 0: return np.inf 
        
        E_z = np.sqrt(om * (1+z)**3 + orad * (1+z)**4 + ol)
        h_lcdm = h0_base * E_z
        gain = 1 + alpha * np.exp(-(z / self.z_edge)**2)
        return h_lcdm * gain

    def calc_theta_star(self, h0_base):
        alpha = (self.target_h0_local / h0_base) - 1.0

        def rs_integrand(z):
            h = h0_base / 100.0
            ob = self.omega_b / h**2
            og = (5.38e-5 * 0.674**2) / h**2
            R = (3.0 * ob) / (4.0 * og * (1+z))
            cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
            return cs / self.get_h_z(z, h0_base, alpha)
        
        rs, _ = quad(rs_integrand, self.z_star, 1e6)
        da, _ = quad(lambda z: 1.0 / self.get_h_z(z, h0_base, alpha), 0, self.z_star)
        
        return (rs / da)

    def objective(self, h0_base):
        return self.calc_theta_star(h0_base) - self.theta_star_target

    def run_optimization(self):
        print("Starting HIA Joint Optimizer (Physical Density Locked)...")
        try:
            res = root_scalar(self.objective, bracket=[50.0, 70.0], method='brentq')
            best_h0_base = res.root
        except ValueError:
            res = minimize_scalar(lambda x: abs(self.objective(x)), bounds=(50.0, 70.0), method='bounded')
            best_h0_base = res.x
            
        best_alpha = (self.target_h0_local / best_h0_base) - 1.0
        final_theta = self.calc_theta_star(best_h0_base)
        precision = abs(final_theta - self.theta_star_target) / self.theta_star_target
        
        return best_h0_base, best_alpha, final_theta, precision

if __name__ == "__main__":
    optimizer = HIA_Optimizer(z_edge=0.5)
    best_h0, best_alpha, final_theta, prec = optimizer.run_optimization()
    
    print("\n==================================================")
    print(" HIA v24.5 Physical Joint Fit Results ")
    print("==================================================")
    print(f"Target Local H0 : {optimizer.target_h0_local:.2f} km/s/Mpc")
    print(f"True Baseline H : {best_h0:.4f} km/s/Mpc")
    print(f"Local Gain Alpha: {best_alpha:.6f} (22.56% burst)")
    print("--------------------------------------------------")
    print(f"CMB 100*Theta_* : {final_theta*100:.6f} (Target: 1.041100)")
    print(f"Residual Error  : {prec:.2e}")

