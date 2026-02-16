"""
RMP Cosmology Validator v4.3 (Official Data Source Edition)
-----------------------------------------------------------
Features: 
- Uses the LATEST Pantheon+ DataRelease (2025/2026 Source)
- MCMC Parameter Estimation (via emcee)
- Strict Academic Integrity: Real Data Only (No Simulations)
- Corrected Redshift-Distance Numerical Integration
"""


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
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import emcee
import corner
import requests
import io
import sys

# --- Constants ---
C_LIGHT = 299792.458  # km/s
H0_PLANCK = 67.36     # Planck 2018 baseline

# --- Core RMP Model ---
def h_rmp_model(z, h0, alpha):
    """RMP v2.0 Damped Projection Model"""
    return H0_PLANCK + (h0 - H0_PLANCK) * (1 / np.cosh(alpha * np.log(1 + z)))

def get_dl_theory(z, h0, alpha):
    """Numerical Integration for Luminosity Distance"""
    integrand = lambda zp: 1.0 / h_rmp_model(zp, h0, alpha)
    res, _ = integrate.quad(integrand, 0, z)
    return (1 + z) * C_LIGHT * res

def mu_theory(z, h0, alpha):
    """Theoretical Distance Modulus"""
    dl = get_dl_theory(z, h0, alpha)
    if dl <= 0: return 1e10
    return 5 * np.log10(dl) + 25

# --- Likelihood Function ---
def log_likelihood(theta, z_data, mu_data, mu_err):
    h0, alpha = theta
    # Physical priors: H0 [60, 85], Alpha [0.1, 3.0]
    if h0 < 60 or h0 > 85 or alpha < 0.1 or alpha > 3.0:
        return -np.inf
    
    # Calculate model predictions
    mu_model = np.array([mu_theory(z, h0, alpha) for z in z_data])
    
    # Chi-squared likelihood (Assuming diagonal covariance for this validator)
    chi2 = np.sum(((mu_data - mu_model) / mu_err)**2)
    return -0.5 * chi2

def run_mcmc_analysis(z_obs, mu_obs, err_obs):
    print("\n[*] Starting MCMC Sampling (emcee)...")
    print("    (This involves numerical integration for each step, please wait...)")
    
    # Initialize 32 walkers around a reasonable starting point
    pos = [73.0, 1.3] + 1e-3 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(z_obs, mu_obs, err_obs))
    
    # Run MCMC: 600 steps (Total samples = 32 * 600 = 19,200)
    sampler.run_mcmc(pos, 600, progress=True)
    
    # Discard burn-in (first 150 steps) and thin the chain
    samples = sampler.get_chain(discard=150, thin=15, flat=True)
    return samples

# --- Data Loading (Latest Official Source) ---
def load_pantheon_data():
    """
    Loads data from the official PantheonPlusSH0ES/DataRelease repository.
    Path: Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat
    """
    urls = [
        # 1. NEW Official DataRelease Repo (The one you provided)
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        # 2. Backup Mirror (Duke University)
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/PantheonPlusSH0ES.github.io/main/Pantheon%2B_Data/v1/Pantheon%2BSH0ES.dat"
    ]
    
    df = None
    for url in urls:
        try:
            print(f"[*] Attempting to fetch real SNe data from: {url[:60]}...")
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                # Use strict parsing to avoid reading HTML error pages as data
                temp_df = pd.read_csv(io.StringIO(r.text), sep=r'\s+', comment='#', engine='python')
                
                # Verify essential columns exist
                if 'zHD' in temp_df.columns and 'MU_SH0ES' in temp_df.columns:
                    df = temp_df
                    print("✅ Successfully loaded REAL Pantheon+ dataset!")
                    break
        except Exception as e:
            print(f"[-] Source failed: {e}")
            continue
    return df

# --- Main Execution ---
def main():
    print("--- RMP Academic Validator v4.3 (Official Source) ---")
    
    # 1. Load REAL data
    df = load_pantheon_data()
    
    if df is None:
        print("\n[❌ CRITICAL ERROR] Could not connect to Pantheon+ DataRelease.")
        print("[!] Academic integrity policy: Simulated data fallback is DISABLED.")
        print("[*] Please manually download 'Pantheon+SH0ES.dat' from:")
        print("    https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR")
        return

    # 2. Pre-process Data 
    # (Sampling 300 points for demonstration speed. For final paper, use all points.)
    print(f"[*] Data size: {len(df)} Supernovae. Sampling 300 points for rapid MCMC...")
    df_sample = df.sample(n=300, random_state=42)
    
    z_obs = df_sample['zHD'].values
    mu_obs = df_sample['MU_SH0ES'].values
    err_obs = df_sample['MU_SH0ES_ERR_DIAG'].values

    # 3. MCMC Analysis
    samples = run_mcmc_analysis(z_obs, mu_obs, err_obs)
    
    # 4. Results Generation
    h0_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    alpha_mcmc = np.percentile(samples[:, 1], [16, 50, 84])
    
    print("\n" + "="*50)
    print(f" FINAL RMP POSTERIOR RESULTS (OFFICIAL REAL DATA)")
    print("="*50)
    print(f" H0    : {h0_mcmc[1]:.2f} (+{h0_mcmc[2]-h0_mcmc[1]:.2f} / -{h0_mcmc[1]-h0_mcmc[0]:.2f}) km/s/Mpc")
    print(f" Alpha : {alpha_mcmc[1]:.3f} (+{alpha_mcmc[2]-alpha_mcmc[1]:.3f} / -{alpha_mcmc[1]-alpha_mcmc[0]:.3f})")
    print("="*50)

    # 5. Visualizations
    try:
        fig = corner.corner(samples, 
                            labels=["$H_0$", r"$\alpha$"], 
                            truths=[h0_mcmc[1], alpha_mcmc[1]],
                            show_titles=True, 
                            title_fmt=".3f")
        plt.savefig("rmp_mcmc_corner_REAL.png")
        print("\n[🎉] Corner Plot saved as 'rmp_mcmc_corner_REAL.png'")
        print("[*] This plot is your primary statistical evidence.")
    except Exception as e:
        print(f"[!] Plotting error (data is fine, just visualization): {e}")

if __name__ == "__main__":
    main()
    

