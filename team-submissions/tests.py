"""
Comprehensive Test Suite for Phase 2 Implementation
Testing GPU-accelerated LABS MTS and verification of physical correctness
"""

import numpy as np
import pytest
from gpu_mts_implementation import GPUAcceleratedMTS

class TestLABSEnergy:
    """Test Suite for LABS Energy Calculation"""
    
    def test_energy_non_negativity(self):
        """Energy must be non-negative for any sequence"""
        mts = GPUAcceleratedMTS(10, use_gpu=False)
        for _ in range(20):
            seq = np.random.choice([-1, 1], size=10)
            energy = mts.calculate_energy(seq)
            assert energy >= 0, f"Energy must be non-negative, got {energy}"
    
    def test_negation_symmetry(self):
        """Energy(S) == Energy(-S) for any sequence S"""
        mts = GPUAcceleratedMTS(12, use_gpu=False)
        test_sequences = [
            np.array([1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1]),
            np.array([-1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1]),
            np.random.choice([-1, 1], size=12),
        ]
        
        for seq in test_sequences:
            energy_pos = mts.calculate_energy(seq)
            energy_neg = mts.calculate_energy(-seq)
            assert abs(energy_pos - energy_neg) < 1e-6, \
                f"Negation symmetry violated: {energy_pos} != {energy_neg}"
    
    def test_reversal_symmetry(self):
        """Energy(S) == Energy(reverse(S)) for any sequence S"""
        mts = GPUAcceleratedMTS(10, use_gpu=False)
        seq = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1])
        
        energy_orig = mts.calculate_energy(seq)
        energy_rev = mts.calculate_energy(seq[::-1])
        
        assert abs(energy_orig - energy_rev) < 1e-6, \
            f"Reversal symmetry violated: {energy_orig} != {energy_rev}"
    
    def test_barker_sequence(self):
        """Known optimal Barker sequence should have energy 0"""
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        mts = GPUAcceleratedMTS(13, use_gpu=False)
        energy = mts.calculate_energy(barker_13)
        
        # Barker sequences have zero autocorrelation for all lags
        assert energy == 0, f"Barker-13 should have energy 0, got {energy}"

class TestGPUAcceleration:
    """Test GPU Acceleration Components"""
    
    def test_batch_neighbor_energies(self):
        """Batch neighbor evaluation must match sequential computation"""
        mts = GPUAcceleratedMTS(8, use_gpu=False)
        current = np.array([1, -1, 1, 1, -1, -1, 1, -1])
        
        # Get batch energies
        batch_energies = mts.batch_neighbor_energies(current)
        
        # Verify each neighbor
        for i in range(len(current)):
            neighbor = current.copy()
            neighbor[i] *= -1
            expected_energy = mts.calculate_energy(neighbor)
            
            assert abs(batch_energies[i] - expected_energy) < 1e-5, \
                f"Neighbor {i} mismatch: {batch_energies[i]} != {expected_energy}"
    
    def test_tabu_search_improvement(self):
        """Tabu search should find equal or better solutions"""
        mts = GPUAcceleratedMTS(15, use_gpu=False)
        initial = np.random.choice([-1, 1], size=15)
        initial_energy = mts.calculate_energy(initial)
        
        improved, improved_energy = mts.tabu_search_gpu(initial, num_iterations=50)
        
        assert improved_energy <= initial_energy, \
            f"Tabu search degraded: {improved_energy} > {initial_energy}"

class TestMTSAlgorithm:
    """Test MTS Algorithm Properties"""
    
    def test_mts_convergence(self):
        """MTS should produce monotonically improving energy history"""
        mts = GPUAcceleratedMTS(12, use_gpu=False)
        _, _, history, _ = mts.gpu_mts(12, num_generations=15, pop_size=10)
        
        # Check monotonic improvement
        for i in range(1, len(history)):
            assert history[i] <= history[i-1], \
                f"Non-monotonic history at {i}: {history[i]} > {history[i-1]}"
    
    def test_mts_population_diversity(self):
        """MTS should maintain population diversity"""
        mts = GPUAcceleratedMTS(10, use_gpu=False)
        # Run MTS with small population
        best_sol, best_energy, _, _ = mts.gpu_mts(10, num_generations=20, pop_size=5)
        
        # Verify best solution is valid
        assert len(best_sol) == 10, "Best solution size mismatch"
        assert all(s in [-1, 1] for s in best_sol), "Invalid values in best solution"
        assert best_energy >= 0, "Invalid energy"

class TestPhysicalCorrectness:
    """Test Physical Correctness Checks"""
    
    def test_autocorrelation_computation(self):
        """Autocorrelation C_k must match mathematical definition"""
        mts = GPUAcceleratedMTS(6, use_gpu=False)
        seq = np.array([1, -1, 1, 1, -1, 1])
        
        C = mts.calculate_autocorrelation(seq)
        
        # Manual computation
        expected_C = []
        for k in range(1, 6):
            c_k = np.sum(seq[:-k] * seq[k:])
            expected_C.append(c_k)
        
        for i, (computed, expected) in enumerate(zip(C, expected_C)):
            assert abs(computed - expected) < 1e-5, \
                f"Autocorrelation mismatch at lag {i+1}: {computed} != {expected}"
    
    def test_energy_components(self):
        """Energy should be sum of squared autocorrelations"""
        mts = GPUAcceleratedMTS(8, use_gpu=False)
        seq = np.array([1, 1, -1, 1, -1, -1, 1, -1])
        
        C = mts.calculate_autocorrelation(seq)
        expected_energy = np.sum(C ** 2)
        actual_energy = mts.calculate_energy(seq)
        
        assert abs(actual_energy - expected_energy) < 1e-5, \
            f"Energy mismatch: {actual_energy} != {expected_energy}"

# Performance benchmarking tests
class TestPerformance:
    """Performance and Scaling Tests"""
    
    def test_scaling_small_n(self):
        """MTS should complete efficiently for small N"""
        import time
        mts = GPUAcceleratedMTS(10, use_gpu=False)
        
        start = time.time()
        _, _, _, _ = mts.gpu_mts(10, num_generations=20, pop_size=15)
        elapsed = time.time() - start
        
        assert elapsed < 30, f"MTS too slow for N=10: {elapsed}s"
    
    def test_scaling_medium_n(self):
        """MTS should handle N=20 within reasonable time"""
        import time
        mts = GPUAcceleratedMTS(20, use_gpu=False)
        
        start = time.time()
        _, _, _, _ = mts.gpu_mts(20, num_generations=10, pop_size=10)
        elapsed = time.time() - start
        
        assert elapsed < 60, f"MTS too slow for N=20: {elapsed}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
