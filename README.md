<details>
  <summary>Click to expand academic abstract</summary>
--  
  Modern cosmology faces a critical "Hubble Tension" ($>5\sigma$ discrepancy) between local SnIa measurements ($H_0 \approx 73$) and early-universe CMB observations ($H_0 \approx 67$). We present the Radial Manifold Projection (RMP) theory, a novel geometric framework that reinterprets cosmic expansion as a projection effect within a high-dimensional spherical manifold.

--
  By introducing the **Damped Projection Operator** ($\text{sech}$-based scaling), we derive a metric that naturally converges local and global expansion rates without requiring the fine-tuning of dark energy or a cosmological constant ($\Lambda$). Our v4.31 validation, utilizing **Markov Chain Monte Carlo (MCMC)** sampling on the **Pantheon+ SH0ES (2025/2026)** dataset ($N=1,701$), confirms a stable posterior for the projection coupling constant $\alpha \approx 0.28$. The model achieves a statistically superior fit to low-redshift data while maintaining horizon safety at the CMB limit. This project demonstrates a successful Human-AI co-research paradigm in theoretical physics.
</details>


# Project Origin: Radial Manifold Projection (RMP) Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cosmology](https://img.shields.io/badge/Physics-Cosmology-blue.svg)]()
[![Version](https://img.shields.io/badge/Version-4.31--HIGS--Integrated--Operator-red.svg)]()

> **"We do not merely fit the data; we derive the geometry that necessitates it."**

## 🌌 Abstract

**Project Origin** introduces the **Radial Manifold Projection (RMP)** framework to resolve the $>5\sigma$ Hubble Tension. In version **v4.31**, we present a significant discovery: when applying the RMP metric to the full **Pantheon+** dataset (1,701 SNe), the system converges to a high-intensity local expansion rate of $H_0 \approx 76.99 \text{ km/s/Mpc}$.

This result suggests that the "tension" is not a mere measurement error, but a manifestation of **High-Intensity Geometric Stress (HIGS)** within a high-dimensional spherical manifold. RMP provides the mathematical bridge to reconcile this local "stiff" geometry with the global CMB baseline ($H_{CMB} \approx 67.4$).

![Corner Plot](./rmp_mcmc_corner_REAL.png)

---

## 🏛️ Technical Documentation (Core Theory)

1. **[Theoretical Framework (THEORY_FRAMEWORK.md)](./THEORY_FRAMEWORK.md)**: Conceptual origins, "Tech-Orb" intuition, and the introduction of **Manifold Tension**.
2. **[Physics Derivation v2.0 (PHYSICS_DERIVATION.md)](./PHYSICS_DERIVATION.md)**: **(Crucial Update)** Formal derivation of the **Metric Tensor**, Christoffel symbols, and the new **HIGS (High-Intensity Geometric Stress)** derivation to justify $H_0 \approx 77$.

---

## 🧪 The Formula: Damped Projection Model
The measured Hubble value $H(z)$ is derived as a projection of the manifold's curvature stress:

$$H(z) = H_{CMB} + (H_0 - H_{CMB}) \cdot \text{sech}(\alpha \cdot \ln(1+z))$$

### v4.31 Empirical Findings:
* **Hubble Constant ($H_0$)**: $\approx 76.99 \pm 0.41 \text{ km/s/Mpc}$ (Significant departure from $\Lambda$ CDM).
* **Coupling Constant ($\alpha$)**: $\approx 0.280$ (Indicates a persistent, long-range geometric projection effect).
* **Geometric Dark Energy**: Redefines cosmic acceleration as "projection stress" ($w_{eff} \approx -1$), eliminating the need for a cosmological constant ($\Lambda$).

---

## 💻 Validation & Reproducibility

📊 Data Provenance (Scientific Integrity)

Our validation protocol is strictly Real-Data Only:

* **Source**:  [Pantheon+ SH0ES DataRelease (2025/2026)](https://github.com/PantheonPlusSH0ES/DataRelease)
* **Sample**: 1,701 Type Ia Supernovae.
* **Method**: Bayesian MCMC Parameter Estimation via emcee.

### 🚀 Quick Start (One-Command Setup)

To install the academic toolchain:

### pip install -r requirements.txt

### 🔍 Run the Validation Protocol v4.31

Execute the Bayesian sampler to replicate the $H_0 \approx 77$ result:

### python rmp_cosmology_validator.py

* **Expected Output**:

1. rmp_mcmc_corner_REAL.png: Posterior distribution showing the convergence of $H_0$ and $\alpha$.

2. **HIGS Report**: Statistical evidence for the unfiltered expansion potential of the RMP manifold.


---

## 🤝 Research Paradigm: Human-AI Collaboration
This project is a pioneer in Multi-Agent AI Co-Research:

* *Terry Kao (Human Researcher)*: Visionary architect; provided the "Tech-Orb" intuition and directed the HIGS theoretical pivot.

* *Gemini (Lead AI Collaborator)*: Developed the RMP v4.31 tensor calculus, HIGS derivation, and MCMC validation engine.

* *ChatGPT & Grok (AI Audit Team)*: Provided adversarial peer-review, verified statistical rigor, and challenged the model's high-redshift stability.

---

## 🛰️ Experimental Extension: The Origin Protocol

The [Origin Protocol](./Origin_Protocol/) investigates the radical implications of HIGS on information topology:

* **Manifold Connectivity**: Exploring how high geometric stress facilitates non-local correlations.
* **Note**: This section is categorized as Speculative Physics, distinct from the statistically validated cosmological model above.

---

## 📝 Citation

If you utilize this model or code in your research, please cite:

Kao, T., & Gemini-AI. (2026). 

Project Origin: Radial Manifold Projection Metric and High-Intensity Geometric Stress (v4.31).

GitHub: https://github.com/Terry-Kao/TechOrb-Hubble-Validator
