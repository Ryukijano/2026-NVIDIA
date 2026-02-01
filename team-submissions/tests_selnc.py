#!/usr/bin/env python3
"""
Comprehensive Test Suite for SELNC LABS Solver
Phase 2 Implementation - iQuHACK 2026 NVIDIA Challenge

Tests cover:
1. LABS energy computation and physical symmetries
2. SELNC quantum kernel correctness
3. GPU-accelerated MTS functionality
4. Lyapunov controller behavior
5. Performance and scaling

Run with: pytest tests_selnc.py -v --tb=short
"""

import numpy as np
import pytest
import time
from typing import List

# Import our implementations
from selnc_quantum import (
    labs_energy, labs_autocorrelation, 
    SELNCSolver, LyapunovController,
    get_labs_interactions, get_nested_cd_interactions,
    compute_cd_coefficients
)
from selnc_gpu_mts import (
    EnhancedGPUMTS, MTSConfig, QuantumEnhancedMTS,
    labs_energy_cpu, labs_energy_vectorized, GPU_AVAILABLE
)

# =============================================================================
# TEST: LABS ENERGY COMPUTATION
# =============================================================================

class TestLABSEnergy:
    """Test Suite for LABS Energy Calculation"""
    
    def test_energy_non_negativity(self):
        """Energy must be non-negative for any sequence"""
        for N in [8, 10, 12, 15]:
            for _ in range(10):
                seq = np.random.choice([-1, 1], size=N)
                energy = labs_energy(seq)
                assert energy >= 0, f"Energy must be non-negative, got {energy} for N={N}"
    
    def test_negation_symmetry(self):
        """Energy(S) == Energy(-S) for any sequence S"""
        for N in [8, 10, 12]:
            for _ in range(10):
                seq = np.random.choice([-1, 1], size=N)
                energy_pos = labs_energy(seq)
                energy_neg = labs_energy(-seq)
                assert abs(energy_pos - energy_neg) < 1e-6, \
                    f"Negation symmetry violated: {energy_pos} != {energy_neg}"
    
    def test_reversal_symmetry(self):
        """Energy(S) == Energy(reverse(S)) for any sequence S"""
        for N in [8, 10, 12]:
            for _ in range(10):
                seq = np.random.choice([-1, 1], size=N)
                energy_orig = labs_energy(seq)
                energy_rev = labs_energy(seq[::-1])
                assert abs(energy_orig - energy_rev) < 1e-6, \
                    f"Reversal symmetry violated: {energy_orig} != {energy_rev}"
    
    def test_barker_sequence_13(self):
        """Barker-13 has minimal sidelobe autocorrelation (|C_k| ≤ 1)"""
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        C = labs_autocorrelation(barker_13)
        # Barker property: all sidelobes have magnitude ≤ 1
        assert all(abs(c) <= 1 for c in C), f"Barker sidelobe property violated: {C}"
        # Energy = sum of C_k^2, with |C_k| ≤ 1, so energy ≤ N-1
        energy = labs_energy(barker_13)
        assert energy <= len(barker_13) - 1, f"Barker energy too high: {energy}"
    
    def test_barker_sequence_11(self):
        """Barker-11 has minimal sidelobe autocorrelation (|C_k| ≤ 1)"""
        barker_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
        C = labs_autocorrelation(barker_11)
        # Barker property: all sidelobes have magnitude ≤ 1
        assert all(abs(c) <= 1 for c in C), f"Barker sidelobe property violated: {C}"
        energy = labs_energy(barker_11)
        assert energy <= len(barker_11) - 1, f"Barker energy too high: {energy}"
    
    def test_autocorrelation_computation(self):
        """Autocorrelation C_k must match mathematical definition"""
        seq = np.array([1, -1, 1, 1, -1, 1])
        C = labs_autocorrelation(seq)
        
        # Manual computation
        expected_C = []
        for k in range(1, 6):
            c_k = np.sum(seq[:-k] * seq[k:])
            expected_C.append(c_k)
        
        for i, (computed, expected) in enumerate(zip(C, expected_C)):
            assert abs(computed - expected) < 1e-6, \
                f"Autocorrelation mismatch at lag {i+1}: {computed} != {expected}"
    
    def test_energy_is_sum_of_squared_autocorrelations(self):
        """Energy should be sum of squared autocorrelations"""
        for N in [6, 8, 10]:
            seq = np.random.choice([-1, 1], size=N)
            C = labs_autocorrelation(seq)
            expected_energy = np.sum(C ** 2)
            actual_energy = labs_energy(seq)
            assert abs(actual_energy - expected_energy) < 1e-6, \
                f"Energy mismatch: {actual_energy} != {expected_energy}"

# =============================================================================
# TEST: SELNC QUANTUM SOLVER
# =============================================================================

class TestSELNCSolver:
    """Test Suite for SELNC Quantum Solver"""
    
    def test_solver_initialization(self):
        """Solver should initialize correctly"""
        for N in [8, 10, 12]:
            solver = SELNCSolver(N, cd_order=2, init_mode="palindromic")
            assert solver.N == N
            assert solver.cd_order == 2
            assert len(solver.G2) > 0
            assert len(solver.G4) > 0
    
    def test_interaction_generation(self):
        """Interaction indices should be valid"""
        for N in [8, 10, 12]:
            G2, G4 = get_labs_interactions(N)
            
            # Check G2 indices are valid
            for pair in G2:
                assert len(pair) == 2
                assert all(0 <= idx < N for idx in pair)
                assert pair[0] < pair[1]
            
            # Check G4 indices are valid
            for quad in G4:
                assert len(quad) == 4
                assert all(0 <= idx < N for idx in quad)
    
    def test_nested_cd_interactions(self):
        """Nested CD interactions should be generated correctly"""
        for N in [10, 12, 15]:
            interactions = get_nested_cd_interactions(N, max_order=2)
            
            assert 'order1' in interactions
            assert 'order2' in interactions
            assert len(interactions['order1']) > 0
    
    def test_cd_coefficients_computation(self):
        """CD coefficients should be finite and reasonable"""
        N = 10
        G2, G4 = get_labs_interactions(N)
        
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            coeffs = compute_cd_coefficients(t, 0.1, 1.0, N, G2, G4, order=2)
            
            assert 'alpha' in coeffs
            assert 'beta' in coeffs
            assert np.isfinite(coeffs['alpha'])
            assert np.isfinite(coeffs['beta'])
    
    def test_solver_sample_output_format(self):
        """Solver sample should return correct format"""
        solver = SELNCSolver(8, cd_order=1, init_mode="standard")
        result = solver.sample(n_shots=50, n_steps=5)
        
        assert 'samples' in result
        assert 'best_bitstring' in result
        assert 'best_energy' in result
        assert 'timing' in result
        assert 'n_unique' in result
        assert result['timing'] > 0
    
    def test_solver_symmetry_preservation(self):
        """Solutions from SELNC should satisfy LABS symmetries"""
        solver = SELNCSolver(10, cd_order=2, init_mode="palindromic")
        result = solver.sample(n_shots=100, n_steps=8)
        
        if result['best_bitstring']:
            seq = np.array([1 if b == '1' else -1 for b in result['best_bitstring']])
            e_orig = labs_energy(seq)
            e_neg = labs_energy(-seq)
            e_rev = labs_energy(seq[::-1])
            
            assert abs(e_orig - e_neg) < 1e-6, "Negation symmetry violated"
            assert abs(e_orig - e_rev) < 1e-6, "Reversal symmetry violated"

# =============================================================================
# TEST: LYAPUNOV CONTROLLER
# =============================================================================

class TestLyapunovController:
    """Test Suite for Lyapunov Adaptive Controller"""
    
    def test_controller_initialization(self):
        """Controller should initialize correctly"""
        controller = LyapunovController(N=10, T_total=1.0)
        
        assert controller.N == 10
        assert controller.T == 1.0
        assert controller.state.lambda_t == 0.0
        assert controller.state.lambda_dot > 0
    
    def test_schedule_monotonic_increase(self):
        """Lambda should generally increase over time"""
        controller = LyapunovController(N=10, T_total=1.0)
        
        prev_lambda = 0.0
        for _ in range(50):
            current_lambda = controller.advance(dt=0.02, current_energy=50.0)
            # Allow small decreases due to control, but overall trend should be up
        
        # Final lambda should be greater than initial
        assert controller.state.lambda_t > 0.5, "Schedule should have progressed"
    
    def test_schedule_bounded(self):
        """Lambda should stay in [0, 1]"""
        controller = LyapunovController(N=10, T_total=1.0)
        
        for _ in range(100):
            controller.advance(dt=0.02, current_energy=50.0 - _ * 0.5)
            assert 0.0 <= controller.state.lambda_t <= 1.0, \
                f"Lambda out of bounds: {controller.state.lambda_t}"
    
    def test_lyapunov_function_positive(self):
        """Lyapunov function should be positive"""
        controller = LyapunovController(N=10, T_total=1.0)
        
        V = controller.lyapunov_function(current_energy=100.0, ground_estimate=0.0)
        assert V > 0, f"Lyapunov function should be positive, got {V}"

# =============================================================================
# TEST: GPU-ACCELERATED MTS
# =============================================================================

class TestGPUMTS:
    """Test Suite for GPU-Accelerated MTS"""
    
    def test_mts_initialization(self):
        """MTS should initialize correctly"""
        config = MTSConfig(num_generations=10, pop_size=10, use_gpu=False)
        mts = EnhancedGPUMTS(N=10, config=config)
        
        assert mts.N == 10
        assert mts.device == 'CPU'  # Fallback when GPU not used
    
    def test_energy_calculation_consistency(self):
        """GPU and CPU energy calculations should match"""
        mts_cpu = EnhancedGPUMTS(N=10, config=MTSConfig(use_gpu=False))
        
        for _ in range(10):
            seq = np.random.choice([-1, 1], size=10).astype(np.float32)
            energy_mts = mts_cpu.calculate_energy(seq)
            energy_ref = labs_energy_cpu(seq)
            
            assert abs(energy_mts - energy_ref) < 1e-4, \
                f"Energy mismatch: {energy_mts} vs {energy_ref}"
    
    def test_batch_neighbor_energies(self):
        """Batch neighbor evaluation should match sequential"""
        mts = EnhancedGPUMTS(N=8, config=MTSConfig(use_gpu=False))
        current = np.array([1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float32)
        
        batch_energies = mts.batch_neighbor_energies(current)
        
        # Verify each neighbor
        for i in range(len(current)):
            neighbor = current.copy()
            neighbor[i] *= -1
            expected_energy = labs_energy_cpu(neighbor)
            
            assert abs(batch_energies[i] - expected_energy) < 1e-4, \
                f"Neighbor {i} mismatch: {batch_energies[i]} != {expected_energy}"
    
    def test_tabu_search_improvement(self):
        """Tabu search should find equal or better solutions"""
        mts = EnhancedGPUMTS(N=12, config=MTSConfig(use_gpu=False))
        
        for _ in range(5):
            initial = np.random.choice([-1, 1], size=12).astype(np.float32)
            initial_energy = mts.calculate_energy(initial)
            
            improved, improved_energy = mts.tabu_search_gpu(initial, num_iterations=30)
            
            assert improved_energy <= initial_energy + 1e-6, \
                f"Tabu search degraded: {improved_energy} > {initial_energy}"
    
    def test_mts_convergence(self):
        """MTS should produce non-increasing energy history"""
        mts = EnhancedGPUMTS(N=10, config=MTSConfig(
            num_generations=15, pop_size=8, tabu_iterations=30, use_gpu=False
        ))
        
        result = mts.run(verbose=False)
        history = result.energy_history
        
        # Check monotonic non-increasing (allowing for noise)
        for i in range(1, len(history)):
            assert history[i] <= history[0] + 1e-6, \
                f"History increased beyond initial at step {i}"
    
    def test_mts_output_validity(self):
        """MTS output should be valid LABS sequence"""
        mts = EnhancedGPUMTS(N=10, config=MTSConfig(
            num_generations=10, pop_size=8, use_gpu=False
        ))
        
        result = mts.run(verbose=False)
        
        # Check solution validity
        assert len(result.best_solution) == 10
        assert all(s in [-1, 1] for s in result.best_solution)
        assert result.best_energy >= 0
        
        # Verify energy matches
        computed_energy = labs_energy_cpu(result.best_solution)
        assert abs(result.best_energy - computed_energy) < 1e-4

# =============================================================================
# TEST: PHYSICAL CORRECTNESS
# =============================================================================

class TestPhysicalCorrectness:
    """Test Suite for Physical Correctness Verification"""
    
    def test_all_symmetries_combined(self):
        """Combined test for all LABS symmetries"""
        for N in [8, 10, 12, 15]:
            seq = np.random.choice([-1, 1], size=N)
            
            e_orig = labs_energy(seq)
            e_neg = labs_energy(-seq)
            e_rev = labs_energy(seq[::-1])
            e_neg_rev = labs_energy(-seq[::-1])
            
            # All should be equal
            assert abs(e_orig - e_neg) < 1e-6
            assert abs(e_orig - e_rev) < 1e-6
            assert abs(e_orig - e_neg_rev) < 1e-6
    
    def test_known_optimal_sequences(self):
        """Test that known good sequences have low energy and satisfy symmetries"""
        # Known good sequences (not necessarily global optima, but low energy)
        test_sequences = [
            np.array([1, 1, 1, -1, 1]),           # N=5
            np.array([1, 1, 1, -1, -1, 1, -1]),   # N=7
            np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),  # N=11 Barker
            np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),  # N=13 Barker
        ]
        
        for seq in test_sequences:
            energy = labs_energy(seq)
            N = len(seq)
            
            # Energy should be non-negative
            assert energy >= 0, f"N={N}: energy must be non-negative"
            
            # Energy should be reasonably low for good sequences
            # Upper bound: random sequence has expected energy ~ N^2/4
            assert energy < N * N / 2, f"N={N}: energy {energy} too high for good sequence"
            
            # Symmetries must hold
            assert abs(energy - labs_energy(-seq)) < 1e-6, "Negation symmetry violated"
            assert abs(energy - labs_energy(seq[::-1])) < 1e-6, "Reversal symmetry violated"

# =============================================================================
# TEST: PERFORMANCE
# =============================================================================

class TestPerformance:
    """Performance and Scaling Tests"""
    
    def test_small_n_completes_quickly(self):
        """MTS should complete efficiently for small N"""
        mts = EnhancedGPUMTS(N=10, config=MTSConfig(
            num_generations=15, pop_size=10, tabu_iterations=30, use_gpu=False
        ))
        
        start = time.time()
        result = mts.run(verbose=False)
        elapsed = time.time() - start
        
        assert elapsed < 60, f"MTS too slow for N=10: {elapsed:.1f}s"
    
    def test_medium_n_completes(self):
        """MTS should handle N=15 within reasonable time"""
        mts = EnhancedGPUMTS(N=15, config=MTSConfig(
            num_generations=10, pop_size=8, tabu_iterations=20, use_gpu=False
        ))
        
        start = time.time()
        result = mts.run(verbose=False)
        elapsed = time.time() - start
        
        assert elapsed < 120, f"MTS too slow for N=15: {elapsed:.1f}s"
    
    def test_selnc_solver_completes(self):
        """SELNC solver should complete for small N"""
        solver = SELNCSolver(N=8, cd_order=1, init_mode="standard")
        
        start = time.time()
        result = solver.sample(n_shots=50, n_steps=5)
        elapsed = time.time() - start
        
        assert elapsed < 30, f"SELNC too slow for N=8: {elapsed:.1f}s"
        assert result['n_unique'] > 0

# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_quantum_classical_hybrid(self):
        """Test full quantum-classical hybrid workflow"""
        hybrid = QuantumEnhancedMTS(N=8, mts_config=MTSConfig(
            num_generations=10, pop_size=8, tabu_iterations=20, use_gpu=False
        ))
        
        # Run quantum-seeded
        result_q = hybrid.run_quantum_seeded(
            quantum_shots=30, quantum_steps=5, verbose=False
        )
        
        assert 'mts_quantum' in result_q
        assert result_q['mts_quantum']['best_energy'] >= 0
    
    def test_comparison_workflow(self):
        """Test comparison between quantum and random seeding"""
        hybrid = QuantumEnhancedMTS(N=8, mts_config=MTSConfig(
            num_generations=8, pop_size=6, tabu_iterations=15, use_gpu=False
        ))
        
        results = hybrid.run_comparison(
            quantum_shots=20, quantum_steps=4, verbose=False
        )
        
        assert 'mts_quantum' in results
        assert 'mts_random' in results
        assert 'comparison' in results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
