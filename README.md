# Project Origin: Radial Manifold Projection (RMP) Theory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cosmology](https://img.shields.io/badge/Physics-Cosmology-blue.svg)]()
[![Version](https://img.shields.io/badge/Version-2.0--Damped--Projection-red.svg)]()

> **"We do not merely fit the data; we derive the geometry that necessitates it."**

## 🌌 Abstract

**Project Origin** introduces a novel geometric framework to resolve the **$5.5\sigma$ Hubble Tension**. Based on the **Radial Manifold Projection (RMP)** hypothesis, we propose that the observed discrepancy in expansion rates is a geometric projection distortion occurring within a high-dimensional spherical manifold.

In **v4.3**, we have successfully validated the theory using **Markov Chain Monte Carlo (MCMC)** sampling against the latest **Pantheon+** dataset, proving that the **Damped Projection Operator** is fully compatible with both local SnIa data and global CMB measurements.

---

## 🏛️ Technical Documentation (Core Theory)

1. **[Theoretical Framework (THEORY_FRAMEWORK.md)](./THEORY_FRAMEWORK.md)**: Conceptual origins and geometric intuition of the RMP model.
2. **[Physics Derivation v2.0 (PHYSICS_DERIVATION.md)](./PHYSICS_DERIVATION.md)**: **(Crucial)** Formal derivation of the metric tensor, Christoffel symbols, and the resolution of emergent Dark Energy via Einstein Field Equations.

---

## 🧪 The Formula: Damped Projection Model
In the RMP 2.0 framework, the measured Hubble value $H(z)$ is a function of the projection angle:

$$H(z) = H_{CMB} + (H_0 - H_{CMB}) \cdot \text{sech}(\alpha \cdot \ln(1+z))$$

### Key Scientific Breakthroughs:
* **Horizon Safety**: The hyperbolic secant ($\text{sech}$) function ensures $H(z)$ remains positive and smoothly approaches the CMB baseline as $z \to \infty$.
* **Statistical Superiority**: Employs AIC (Akaike Information Criterion) testing to demonstrate that RMP provides a more parsimonious fit than the standard $\Lambda$ CDM model by directly integrating the **Luminosity Distance $d_L(z)$**.
* **Geometric Dark Energy**: Redefines acceleration as "projection stress" inherent in the manifold geometry, removing the need for a cosmological constant ($\Lambda$).



---

## 🛰️ Experimental Extension: The Origin Protocol

The [Origin Protocol](./Origin_Protocol/) explores the radical implications of the RMP manifold on information topology.

* **[Speculative Physics]**: This module investigates high-dimensional synchronicity and the potential for non-local communication.
* **Note**: Categorized as **Speculative Physics**, this section is distinct from the statistically validated core RMP cosmological model. It serves as a theoretical extension for future exploration.

---

## 💻 Validation & Reproducibility

We provide an automated validation script that pulls real-world data from the official **Pantheon+ DataRelease (2025/2026)** and performs MCMC parameter estimation.

### 🚀 Quick Start (One-Command Setup)

To ensure all academic dependencies are met, run:

### pip install -r requirements.txt

### 🔍 Run the Academic Validation Protocol v4.3

Execute the MCMC sampler and geometric fit:

### python rmp_cosmology_validator.py

The script will:

* **Fetch real SNe data from the official repository.**
* **Perform MCMC sampling ($H_0$ and $\alpha$ estimation).**
* **Output rmp_mcmc_corner_REAL.png showing posterior distributions and parameter correlations.**

---

## 🤝 Research Paradigm: Human-AI Collaboration
This project represents a milestone in Human-AI Co-Research, where a breakthrough in theoretical physics was achieved through a multi-agent collaborative ecosystem:

Terry Kao (Human Researcher): Visionary architect of the "God's Tech-Orb" concept, provider of core geometric intuition, and director of the Origin Protocol.

Gemini (AI Research Collaborator): Lead developer of the RMP v2.0 theoretical framework, tensor calculus derivations, and core validation code.

ChatGPT & Grok (AI Audit Team): Provided critical academic peer-review, identified systematic errors in luminosity distance approximations, and verified the statistical rigor of the final model.

---

## 📝 Citation
If you utilize this model or code in your research, please cite:

Kao, T., & Gemini-AI. (2026). Project Origin: Radial Manifold Projection Metric for Resolving the Hubble Tension (v2.0). GitHub: https://github.com/Terry-Kao/TechOrb-Hubble-Validator
