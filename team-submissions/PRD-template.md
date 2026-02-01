# Product Requirements Document (PRD)

**Project Name:** SELNC | **Team Name:** [Cudits] | **GitHub Repository:** https://github.com/Ryukijano/2026-NVIDIA

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle |
|---|---|---|---|
| **Project Lead** (Architect) | Gyanateet Dutta | [@ryukijano] | [@ryukijano] |
| **GPU Acceleration PIC** (Builder) | Gyanateet Dutta | [@ryukijano] | [@ryukijano] |
| **Quality Assurance PIC** (Verifier) | Gyanateet Dutta | [@ryukijano] | [@ryukijano] |
| **Technical Marketing PIC** (Storyteller) | Gyanateet Dutta | [@ryukijano] | [@ryukijano] |

## 2. The Architecture

**Owner:** Gyanateet Dutta

### Choice of Quantum Algorithm

* • **Algorithm:** Symmetry-Enhanced Lyapunov-Controlled Nested Counterdiabatic (SELNC) Optimization.
* • **Motivation:**
    * ◦ DCQO provides efficient quantum state preparation with significantly reduced circuit depth (6x reduction compared to QAOA) through counterdiabatic driving.
    * ◦ The SELNC approach incorporates higher-order nested commutator terms to better approximate the adiabatic gauge potential, specifically targeting the LABS autocorrelation structure.
    * ◦ By exploiting reversal and negation symmetries, the algorithm operates in a reduced Hilbert space, enhancing sampling efficiency and solution quality.
    * ◦ Adaptive Lyapunov control optimizes the annealing schedule in real-time, preventing transitions out of the ground state at avoided crossings.

### Literature Review

* • **Reference:** Gomez Cadavid, A., Chandarana, P., et al. \"Scaling advantage with quantum-enhanced memetic tabu search for LABS\" (2025). [Link to arXiv:2511.04553v1]
* • **Relevance:** Foundation for quantum-enhanced hybrid workflows for LABS problems.
* • **Reference:** Passarelli, G., et al. \"Counterdiabatic driving in the quantum annealing of the p-spin model.\" Physical Review Research (2020).
* • **Relevance:** Provides theoretical basis for nested commutator expansions in counterdiabatic driving.

## 3. The Acceleration Strategy

**Owner:** Gyanateet Dutta

### Quantum Acceleration (CUDA-Q)

* • **Strategy:**
    * ◦ Utilize CUDA-Q Multi-GPU backends (`nvidia-mgpu`) for state vector simulation of nested counterdiabatic circuits.
    * ◦ Implement batch processing of quantum samples to maximize GPU throughput during the seeding phase.
    * ◦ Use tensor network simulation backends for problem sizes N > 30 to manage memory constraints efficiently.

### Classical Acceleration (MTS)

* • **Strategy:**
    * ◦ Port the Memetic Tabu Search (MTS) to NVIDIA GPUs using CuPy for vectorized energy calculations.
    * ◦ Implement population-parallel local search: each CUDA thread manages a tabu search for a single population member.
    * ◦ Accelerate neighbor exploration by batch-evaluating all single-bit flips in a single GPU kernel call.

### Hardware Targets

* • **Dev Environment:** Google Colab (T4 GPU) for initial testing.
* • **Production Environment:** Brev.dev L4/A100 instances for heavy benchmarking and N=30+ scaling studies.

## 4. The Verification Plan

**Owner:** Gyanateet Dutta

### Unit Testing Strategy

* • **Framework:** Automated pytest suite (`tests.py`) running on each commit.
* • **Automated Assertions:**
    * ◦ `assert calculate_energy(S) == calculate_energy(-S)` (Negation symmetry).
    * ◦ `assert calculate_energy(S) == calculate_energy(S[::-1])` (Reversal symmetry).
    * ◦ `assert calculate_energy(Barker13) == 0` (Known optimal).

## 5. Execution Strategy & Success Metrics

**Owner:** Gyanateet Dutta

### Agentic Workflow

* • **Plan:**
    * ◦ **Lead Agent:** Orchestrates implementation and PRD updates.
    * ◦ **QA Agent:** Generates and runs unit tests for every new kernel.
    * ◦ **Perf Agent:** Profiles GPU kernels and identifies bottlenecks in MTS.

### Success Metrics

* • **Approximation Ratio:** Target > 0.90 for N=25.
* • **Speedup:** Aim for 50x reduction in MTS runtime using GPU acceleration compared to CPU baseline.
* • **Scale:** Successfully solve N=35 problem instances within 10 minutes of total workflow time.

## 6. Resource Management Plan

**Owner:** Gyanateet Dutta

* • **L4 Testing:** 5 hours dev/test ($5.00).
* • **A100 Benchmarking:** 4 hours high-N runs ($8.00).
* • **Buffer:** $7.00 for retries and optimization.
* • **Monitoring:** Alarms set for 30-minute intervals; mandatory instance termination after each benchmarking session.
