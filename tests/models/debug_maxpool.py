"""Debug script to check max-pooling behavior."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool

# Create a simple 4x4 lattice for easier debugging
N = 4
num_nodes = N * N  # 16 nodes

# Create edge_index for 4-connected grid
edges = []
for i in range(N):
    for j in range(N):
        node_idx = i * N + j
        if j < N - 1:  # Right
            edges.append([node_idx, node_idx + 1])
        if i < N - 1:  # Down
            edges.append([node_idx, node_idx + N])

edge_index = torch.tensor(edges, dtype=torch.long).t()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Bidirectional

print(f"Graph: {N}x{N} lattice with {num_nodes} nodes")
print(f"Edge index shape: {edge_index.shape}")

# Create node features (just node indices)
x = torch.arange(num_nodes, dtype=torch.float32).reshape(1, 1, num_nodes, 1)
print(f"\nNode features (original):")
print(x[0, 0, :, 0].reshape(N, N))

# Create strided mask (d=2): select nodes where i%2==0 and j%2==0
active_mask = torch.zeros(1, num_nodes, dtype=torch.bool)
for i in range(N):
    for j in range(N):
        if i % 2 == 0 and j % 2 == 0:
            node_idx = i * N + j
            active_mask[0, node_idx] = True

print(f"\nActive mask (d=2 stride):")
print(active_mask[0].reshape(N, N).int())
print(f"Active nodes: {active_mask.sum().item()} / {num_nodes}")
active_indices = torch.where(active_mask[0])[0]
print(f"Active node indices: {active_indices.tolist()}")

# Apply StridedGraphMaxPool with gamma=2, K=1 (simpler for debugging)
gamma = 2
K = 1
pool = StridedGraphMaxPool(gamma=gamma, K=K, selection_method='stride')

print(f"\n=== Applying StridedGraphMaxPool (γ={gamma}, K={K}) ===")

# Manually check the adjacency for node 0
print(f"\nChecking adjacency for node 0 at position (0,0):")
print(f"  0-hop: node 0 (itself)")
print(f"  2-hop: nodes at distance 2 (e.g., (0,2)=2, (2,0)=8, (1,1)=5)")

x_pooled, new_active_mask, _ = pool(x, edge_index, active_mask=active_mask)

print(f"\nPooled features:")
print(x_pooled[0, 0, :, 0].reshape(N, N))

print(f"\nNew active mask after stride selection:")
print(new_active_mask[0].reshape(N, N).int())
print(f"Active nodes after pooling: {new_active_mask.sum().item()} / {num_nodes}")

# Check specific nodes
print(f"\n=== Checking specific nodes ===")
for node_idx in [0, 2, 8, 10]:
    i, j = node_idx // N, node_idx % N
    was_active = active_mask[0, node_idx].item()
    pooled_val = x_pooled[0, 0, node_idx, 0].item()
    is_active_after = new_active_mask[0, node_idx].item()
    print(f"Node {node_idx} at ({i},{j}): "
          f"initially_active={was_active}, "
          f"pooled_value={pooled_val:.1f}, "
          f"active_after={is_active_after}")

print(f"\n=== Expected for node 0 ===")
print(f"Node 0 at (0,0) should aggregate from:")
print(f"  - Node 0 (0-hop): value 0 (active)")
print(f"  - Node 2 at (0,2) (2-hop): value 2 (active)")  
print(f"  - Node 8 at (2,0) (2-hop): value 8 (active)")
print(f"  - Node 5 at (1,1) (2-hop): value 5 (INACTIVE, should be -inf, ignored)")
print(f"Expected max: 8")
print(f"Actual: {x_pooled[0, 0, 0, 0].item():.1f}")
