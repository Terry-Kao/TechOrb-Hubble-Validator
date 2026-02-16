"""
RMP Cosmology Validator v4.2 (Strict Academic Edition)
-----------------------------------------------
Features: 
- MCMC Parameter Estimation (via emcee)
- REAL-DATA ONLY: Pantheon+ SNe (Strict enforcement)
- Corrected Redshift-Distance Numerical Integration
- No Mock/Simulated Data Fallback for Integrity
"""

!pip install emcee corner
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
    if h0 < 60 or h0 > 85 or alpha < 0.1 or alpha > 3.0:
        return -np.inf
    
    # SNe Likelihood calculation
    mu_model = np.array([mu_theory(z, h0, alpha) for z in z_data])
    chi2 = np.sum(((mu_data - mu_model) / mu_err)**2)
    return -0.5 * chi2

def run_mcmc_analysis(z_obs, mu_obs, err_obs):
    print("\n[*] Starting MCMC Sampling (emcee)... This involves heavy integration.")
    # Initialize walkers around a reasonable starting point
    pos = [73.0, 1.2] + 1e-3 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(z_obs, mu_obs, err_obs))
    # Running 600 steps for better convergence
    sampler.run_mcmc(pos, 600, progress=True)
    
    samples = sampler.get_chain(discard=150, thin=15, flat=True)
    return samples

# --- Data Loading (Real Only) ---
def load_pantheon_data():
    # URLs to Pantheon+ official/mirror repositories
    urls = [
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/PantheonPlusSH0ES.github.io/main/Pantheon%2B_Data/v1/Pantheon%2BSH0ES.dat",
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/PantheonPlus/main/data/Pantheon%2B_Data/v1/Pantheon%2BSH0ES.dat"
    ]
    
    df = None
    for url in urls:
        try:
            print(f"[*] Attempting to fetch real SNe data: {url[:50]}...")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                # Validating data headers
                temp_df = pd.read_csv(io.StringIO(r.text), sep=r'\s+', comment='#', engine='python')
                if 'zHD' in temp_df.columns:
                    df = temp_df
                    print("✅ Successfully loaded real Pantheon+ dataset!")
                    break
        except Exception as e:
            print(f"[-] Mirror failed: {e}")
            continue
    return df

# --- Main Execution ---
def main():
    print("--- RMP Academic Validator v4.2 (Strict Mode) ---")
    
    # 1. Load REAL data only
    df = load_pantheon_data()
    
    if df is None:
        print("\n[❌ CRITICAL ERROR] Could not connect to any real data source (404/Timeout).")
        print("[!] Academic integrity policy: Simulated data fallback is DISABLED.")
        print("[*] Please manually download 'Pantheon+SH0ES.dat' and place it in the project root.")
        print("    Download link: https://github.com/PantheonPlusSH0ES/PantheonPlusSH0ES.github.io/blob/main/Pantheon%2B_Data/v1/Pantheon%2BSH0ES.dat")
        return

    # 2. Pre-process Data (Sampling 400 points for speed in demonstration, use all for final paper)
    df_sample = df.sample(n=400, random_state=42)
    z_obs = df_sample['zHD'].values
    mu_obs = df_sample['MU_SH0ES'].values
    err_obs = df_sample['MU_SH0ES_ERR_DIAG'].values

    # 3. MCMC Analysis
    samples = run_mcmc_analysis(z_obs, mu_obs, err_obs)
    
    # 4. Results Generation
    h0_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    alpha_mcmc = np.percentile(samples[:, 1], [16, 50, 84])
    
    print("\n" + "="*40)
    print(f" FINAL RMP POSTERIOR RESULTS (REAL DATA)")
    print("="*40)
    print(f" H0    : {h0_mcmc[1]:.2f} (+{h0_mcmc[2]-h0_mcmc[1]:.2f} / -{h0_mcmc[1]-h0_mcmc[0]:.2f}) km/s/Mpc")
    print(f" Alpha : {alpha_mcmc[1]:.3f} (+{alpha_mcmc[2]-alpha_mcmc[1]:.3f} / -{alpha_mcmc[1]-alpha_mcmc[0]:.3f})")
    print("="*40)

    # 5. Visualizations
    fig = corner.corner(samples, 
                        labels=["$H_0$", r"$\alpha$"], 
                        truths=[h0_mcmc[1], alpha_mcmc[1]],
                        show_titles=True, 
                        title_fmt=".3f")
    plt.savefig("rmp_mcmc_corner_REAL.png")
    print("\n[🎉] Corner Plot saved as 'rmp_mcmc_corner_REAL.png'")

if __name__ == "__main__":
    main()
    
