# Product Requirements Document (PRD)

**Project Name:** LABS-Solv-V1 | **Team Name:** [Your Team Name] | **GitHub Repository:** https://github.com/Ryukijano/2026-NVIDIA

---

> **Note to Students:** The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist.
>
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.
> * **Goal:** The objective is to convince the reader that you have a solid plan, not just to fill in boxes.

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle |
|---|---|---|---|
| **Project Lead** (Architect) | [Name] | [@handle] | [@handle] |
| **GPU Acceleration PIC** (Builder) | [Name] | [@handle] | [@handle] |
| **Quality Assurance PIC** (Verifier) | [Name] | [@handle] | [@handle] |
| **Technical Marketing PIC** (Storyteller) | [Name] | [@handle] | [@handle] |

---

## 2. The Architecture

**Owner:** Project Lead

### Choice of Quantum Algorithm

* **Algorithm:** Digitized Counterdiabatic Quantum Optimization (DCQO) for quantum seeding, integrated with Memetic Tabu Search (MTS) as a hybrid quantum-classical approach.
  
* **Motivation:** 
  * DCQO provides efficient quantum state preparation with significantly reduced circuit depth (6x reduction compared to QAOA) through counterdiabatic driving
  * The hybrid approach leverages quantum sampling for global exploration while MTS provides local refinement
  * This strategy achieves superior scaling: QE-MTS demonstrates O(1.24^N) scaling vs. classical MTS O(1.34^N) - a measurable quantum advantage for the LABS problem

### Literature Review

* **Reference:** Gomez Cadavid, A., Chandarana, P., et al. "Scaling advantage with quantum-enhanced memetic tabu search for LABS" (2025). [Link to arXiv:2511.04553v1]
* **Relevance:** Directly demonstrates quantum-enhanced optimization for LABS with empirical scaling advantage; validates our hybrid approach and provides theoretical bounds

* **Reference:** Packebusch, T., & Mertens, S. "Low autocorrelation binary sequences." Journal of Physics A: Mathematical and Theoretical (2016)
* **Relevance:** Provides mathematical foundations for LABS energy landscape structure and optimal sequence properties

---

## 3. The Acceleration Strategy

**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)

* **Strategy:** 
  * Use CUDA-Q for efficient DCQO circuit simulation with GPU acceleration
  * Leverage Trotterization of counterdiabatic Hamiltonian with efficient two-qubit and four-qubit gate decompositions
  * Deploy across GPU architectures (L4 for development, A100/H200 for production benchmarks)
  * Circuit depth: ~236K entangling gates for N=37 (vs. 1.4M for QAOA 12-layer)

### Classical Acceleration (MTS)

* **Strategy:** 
  * Implement population-based search with tabu tenure 5-15 for solution refinement
  * Use Quantum-Enhanced seeding: seed MTS population with best bitstrings from DCQO samples
  * Parameters: population size K=100, crossover probability p_comb=0.9, mutation rate p_mut=1/N
  * Measure time-to-solution (TTS) as primary performance metric

### Hardware Targets

* **Dev Environment:** Google Colab with T4 GPU for initial testing and validation
* **Production Environment:** AWS EC2 instances - H200 GPU (2TB RAM) for N=37 final benchmarks, extending to N=40+ targets

---

## 4. The Verification Plan

**Owner:** Quality Assurance PIC

### Unit Testing Strategy

* **Framework:** pytest with numpy-based validation functions
* **AI Hallucination Guardrails:** 
  * All energy calculations verified against known theoretical bounds: E(s) ∈ [0, N(N-1)(N+1)/6]
  * Cross-validation: quantum circuit outputs compared against classical energy function for all small N tests
  * Property-based testing: ensures energy calculations remain consistent across sequence transformations

### Core Correctness Checks

* **Check 1 (Symmetry):** 
  * LABS sequence S and its negation −S must have identical energies
  * Reversal symmetry: sequence S and reverse(S) must have identical energies
  * Assertion: `assert energy(S) == energy(-S)` and `assert energy(S) == energy(reverse(S))`

* **Check 2 (Ground Truth):** 
  * N=2: sequence [1, -1] → E=1.0 (verified)
  * N=4: sequence [1, 1, -1, 1] → E=2.0 (verified)
  * N=13 (Barker sequence): [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1] → E=0 (known optimal)
  * Test suite assertions verify GPU kernel returns exact values for these known cases

---

## 5. Execution Strategy & Success Metrics

**Owner:** Technical Marketing PIC

### Agentic Workflow

* **Plan:** 
  * Colab Notebook as primary development environment with integrated validation cells
  * Self-validation module with 6 comprehensive test suites:
    - TEST 1: Hand calculations verification (N=2,4,8)
    - TEST 2: Symmetry properties (reversal & negation)
    - TEST 3: Energy function properties (non-negativity, Barker sequences)
    - TEST 4: MTS performance validation
    - TEST 5: Quantum-enhanced population seeding analysis
    - TEST 6: Full visualization suite
  * Continuous integration: validation runs on each code modification

### Success Metrics

* **Metric 1 (Approximation):** Target approximation ratio > 0.85 for N=30 (measuring quantum-enhanced solution quality vs. random baseline)
* **Metric 2 (Speedup):** 1.24x-1.37x speedup: demonstrable scaling advantage of QE-MTS (O(1.24^N)) vs. classical MTS (O(1.37^N))
* **Metric 3 (Scale):** Successfully execute and benchmark up to N=37 with full validation

### Visualization Plan

* **Plot 1 - Energy Landscape Distribution:** Histogram showing energy distribution across multiple sequence lengths (N=4,6,8,10,12,14) demonstrating landscape ruggedness and scaling properties
  
* **Plot 2 - Merit Factor Scaling:** Best merit factor F(N) vs. sequence length N, showing theoretical limit (Golay-Rudin bound: F≈12.32) vs. sampled solutions
  
* **Plot 3 - Convergence Comparison:** MTS (random initialization) vs. QE-MTS (quantum seeded) showing convergence curves and quantified speedup advantage

---

## 6. Resource Management Plan

**Owner:** GPU Acceleration PIC

* **Plan:** 
  * Development Phase: Utilize Google Colab free tier (T4 GPU) for all unit tests and small N validations (N≤15)
  * Scaling Phase: Minimal GPU credits for intermediate benchmarks (N=20-30) on cheap L4 instances
  * Final Benchmarking: Concentrated GPU usage (H200 instances) for final 2-4 hours to benchmark N=35-37 and validate scaling predictions
  * Auto-shutdown protocol: Set instance timeout limits to prevent credit waste during breaks
  * Estimated total cost: < 50 GPU-hours for complete validation and benchmarking

* **Validation Checkpoints:** Run full validation suite at each phase transition before proceeding to next GPU tier
