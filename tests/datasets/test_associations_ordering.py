"""
Test to verify correct handling of non-diagonal association matrices.

This test ensures that when TX i is paired with RX j (where i ≠ j),
the signal and interference are computed correctly.
"""

import numpy as np
import torch


def test_association_ordering():
    """Test that association matrix ordering is handled correctly."""
    
    # Simple test case: 3 TXs, 3 RXs with non-diagonal pairing
    m, n = 3, 3
    
    # Non-diagonal associations:
    # TX 0 → RX 2
    # TX 1 → RX 0
    # TX 2 → RX 1
    associations = np.array([
        [0, 0, 1],  # TX 0 serves RX 2
        [1, 0, 0],  # TX 1 serves RX 0
        [0, 1, 0],  # TX 2 serves RX 1
    ], dtype=float)
    
    # Channel gains: h_t[i,j] = channel from TX i to RX j
    # Use distinct values to track which is which
    h_t = np.array([
        [10, 11, 12],  # TX 0 → [RX 0, RX 1, RX 2]
        [20, 21, 22],  # TX 1 → [RX 0, RX 1, RX 2]
        [30, 31, 32],  # TX 2 → [RX 0, RX 1, RX 2]
    ], dtype=float)
    
    print("="*60)
    print("TEST: Association Ordering")
    print("="*60)
    print("\nAssociations (TX i serves RX j if [i,j]=1):")
    print(associations)
    print("\nChannel gains h_t[i,j] (TX i → RX j):")
    print(h_t)
    
    # Build adjacency matrix - ORIGINAL (BUGGY) VERSION
    h_adj_buggy = np.zeros((m + n, m + n))
    h_adj_buggy[:m, m:] = associations * h_t
    h_adj_buggy[m:, :m] = (1 - associations) * h_t.T  # BUG!
    
    # Build adjacency matrix - CORRECTED VERSION
    h_adj_correct = np.zeros((m + n, m + n))
    h_adj_correct[:m, m:] = associations * h_t
    h_adj_correct[m:, :m] = ((1 - associations) * h_t).T  # CORRECT!
    
    print("\n" + "-"*60)
    print("BUGGY VERSION:")
    print("-"*60)
    print("\nTX→RX block (signal - should be correct):")
    print(h_adj_buggy[:m, m:])
    print("\nExpected signal for each RX:")
    print("  RX 0: should get 20 from TX 1")
    print("  RX 1: should get 31 from TX 2")
    print("  RX 2: should get 12 from TX 0")
    
    print("\nRX←TX block (interference - BUGGY!):")
    print(h_adj_buggy[m:, :m])
    print("\nWhat we want for interference:")
    print("  RX 0 ← [TX 0=10, TX 1=0, TX 2=30]  (exclude serving TX 1)")
    print("  RX 1 ← [TX 0=11, TX 1=21, TX 2=0]  (exclude serving TX 2)")
    print("  RX 2 ← [TX 0=0, TX 1=22, TX 2=32]  (exclude serving TX 0)")
    
    print("\n" + "-"*60)
    print("CORRECTED VERSION:")
    print("-"*60)
    print("\nTX→RX block (signal):")
    print(h_adj_correct[:m, m:])
    
    print("\nRX←TX block (interference - CORRECT!):")
    print(h_adj_correct[m:, :m])
    
    # Verify correctness
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    
    # Check signal block (should be same for both)
    assert np.allclose(h_adj_buggy[:m, m:], h_adj_correct[:m, m:]), "Signal block should be identical"
    print("✓ Signal blocks match")
    
    # Check specific entries in interference block
    # RX 0 is served by TX 1, so should have interference from TX 0 and TX 2
    expected_interference_rx0 = [h_t[0, 0], 0, h_t[2, 0]]  # [10, 0, 30]
    actual_interference_rx0 = h_adj_correct[m+0, :m]
    
    print(f"\nRX 0 interference:")
    print(f"  Expected: {expected_interference_rx0}")
    print(f"  Actual (correct): {actual_interference_rx0.tolist()}")
    print(f"  Actual (buggy):   {h_adj_buggy[m+0, :m].tolist()}")
    
    assert np.allclose(actual_interference_rx0, expected_interference_rx0), "RX 0 interference incorrect"
    print("  ✓ Correct version matches expected")
    
    # RX 1 is served by TX 2, so should have interference from TX 0 and TX 1
    expected_interference_rx1 = [h_t[0, 1], h_t[1, 1], 0]  # [11, 21, 0]
    actual_interference_rx1 = h_adj_correct[m+1, :m]
    
    print(f"\nRX 1 interference:")
    print(f"  Expected: {expected_interference_rx1}")
    print(f"  Actual (correct): {actual_interference_rx1.tolist()}")
    print(f"  Actual (buggy):   {h_adj_buggy[m+1, :m].tolist()}")
    
    assert np.allclose(actual_interference_rx1, expected_interference_rx1), "RX 1 interference incorrect"
    print("  ✓ Correct version matches expected")
    
    # RX 2 is served by TX 0, so should have interference from TX 1 and TX 2
    expected_interference_rx2 = [0, h_t[1, 2], h_t[2, 2]]  # [0, 22, 32]
    actual_interference_rx2 = h_adj_correct[m+2, :m]
    
    print(f"\nRX 2 interference:")
    print(f"  Expected: {expected_interference_rx2}")
    print(f"  Actual (correct): {actual_interference_rx2.tolist()}")
    print(f"  Actual (buggy):   {h_adj_buggy[m+2, :m].tolist()}")
    
    assert np.allclose(actual_interference_rx2, expected_interference_rx2), "RX 2 interference incorrect"
    print("  ✓ Correct version matches expected")
    
    # Show the bug
    if not np.allclose(h_adj_buggy[m:, :m], h_adj_correct[m:, :m]):
        print("\n" + "!"*60)
        print("BUG CONFIRMED:")
        print("!"*60)
        print("The buggy version produces incorrect interference!")
        print(f"Max error: {np.max(np.abs(h_adj_buggy[m:, :m] - h_adj_correct[m:, :m])):.2f}")
    
    print("\n" + "="*60)
    print("✓ TEST PASSED: Corrected version handles associations correctly!")
    print("="*60)


if __name__ == "__main__":
    test_association_ordering()
