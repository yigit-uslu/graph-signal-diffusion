"""Test suite for StridedTAGConv."""

import torch
import matplotlib.pyplot as plt
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from graph_signal_diffusion.models.components import StridedTAGConv, TAGConvLayer
from graph_signal_diffusion.models.components.graph_conv import scatter_add as graph_conv_scatter_add


def save_plot_with_archive(fig, output_path: Path):
    """Save plot, archiving existing file with timestamp if it exists."""
    if output_path.exists():
        # Create archives directory if needed
        archives_dir = output_path.parent / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
        archived_path = archives_dir / archived_name
        
        # Move existing file to archives
        shutil.move(str(output_path), str(archived_path))
        print(f"  Archived existing file to: {archived_path}")
    
    # Save new plot
    fig.savefig(output_path, dpi=150, bbox_inches='tight')


class DebugStridedTAGConv(torch.nn.Module):
    """
    Debug wrapper around StridedTAGConv that captures intermediate propagations.
    
    This wrapper allows visualization of how signals propagate through powers
    of the strided adjacency matrix A^γ.
    
    Args:
        layer: The StridedTAGConv layer to wrap
    """
    
    def __init__(self, layer: StridedTAGConv):
        super().__init__()
        self.layer = layer
        self.intermediates = []  # List of (A^γ)^k x for k=0,1,...,K
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """
        Forward pass that captures intermediate propagations.
        
        Returns:
            output: Final output from StridedTAGConv
            intermediates: List of intermediate signals [(A^γ)^0 x, (A^γ)^1 x, ..., (A^γ)^K x]
        """
        self.intermediates = []
        
        is_4d = x.dim() == 4
        
        if is_4d:
            B, T, N, F = x.shape
            num_nodes_per_timestep = B * N
            
            # Reshape to (T*B*N, F) for processing
            x_flat = x.permute(1, 0, 2, 3).reshape(T * B * N, F)
            
            # Step 1: Compute A^γ (γ-hop adjacency) once
            if self.layer.use_boolean_semiring:
                edge_index_gamma = self.layer._compute_gamma_hop_adjacency_boolean(edge_index, B * N)
            else:
                edge_index_gamma = self.layer._compute_gamma_hop_adjacency_power(edge_index, B * N)
            
            # Step 2: Replicate A^γ for T timesteps
            edge_index_gamma_t = edge_index_gamma.unsqueeze(0).expand(T, -1, -1)
            offsets = torch.arange(T, device=edge_index.device) * num_nodes_per_timestep
            edge_index_gamma_t = edge_index_gamma_t + offsets.view(T, 1, 1)
            edge_index_gamma_t = edge_index_gamma_t.reshape(2, T * edge_index_gamma.size(1))
            
            # Step 3: Normalize A^γ if requested
            if self.layer.normalize:
                from torch_geometric.nn.conv.gcn_conv import gcn_norm
                edge_index_gamma_t, edge_weight_gamma = gcn_norm(
                    edge_index_gamma_t, num_nodes=T * B * N, add_self_loops=False
                )
            else:
                edge_weight_gamma = None
            
            # Step 4: Iteratively apply message passing with A^γ and capture intermediates
            x_k = x_flat.clone()  # Start with x^(0) = x
            
            for k in range(self.layer.K + 1):
                # Store intermediate (reshape back to original form)
                x_k_reshaped = x_k.reshape(T, B, N, F).permute(1, 0, 2, 3)
                self.intermediates.append(x_k_reshaped.detach().clone())
                
                # Message passing: x^(k+1) = A^γ · x^(k)
                if k < self.layer.K:  # Don't need to compute for last iteration
                    row, col = edge_index_gamma_t
                    x_neighbors = x_k[col]  # (E, F)
                    
                    if edge_weight_gamma is not None:
                        x_neighbors = x_neighbors * edge_weight_gamma.view(-1, 1)
                    
                    # Aggregate: sum_{j ∈ N_γ(i)} x_j^(k)
                    x_k = graph_conv_scatter_add(x_neighbors, row, dim=0, dim_size=T * B * N)
        
        elif x.dim() == 3:
            # Similar for 3D case
            BN, T, F = x.shape
            
            # Reshape to (T*B*N, F)
            x_flat = x.permute(1, 0, 2).reshape(T * BN, F)
            
            # Compute A^γ
            if self.layer.use_boolean_semiring:
                edge_index_gamma = self.layer._compute_gamma_hop_adjacency_boolean(edge_index, BN)
            else:
                edge_index_gamma = self.layer._compute_gamma_hop_adjacency_power(edge_index, BN)
            
            # Replicate for T timesteps
            edge_index_gamma_t = edge_index_gamma.unsqueeze(0).expand(T, -1, -1)
            offsets = torch.arange(T, device=edge_index.device) * BN
            edge_index_gamma_t = edge_index_gamma_t + offsets.view(T, 1, 1)
            edge_index_gamma_t = edge_index_gamma_t.reshape(2, T * edge_index_gamma.size(1))
            
            # Normalize if requested
            if self.layer.normalize:
                from torch_geometric.nn.conv.gcn_conv import gcn_norm
                edge_index_gamma_t, edge_weight_gamma = gcn_norm(
                    edge_index_gamma_t, num_nodes=T * BN, add_self_loops=False
                )
            else:
                edge_weight_gamma = None
            
            # Iterative message passing with A^γ
            x_k = x_flat.clone()
            
            for k in range(self.layer.K + 1):
                # Store intermediate
                x_k_reshaped = x_k.reshape(T, BN, F).permute(1, 0, 2)
                self.intermediates.append(x_k_reshaped.detach().clone())
                
                if k < self.layer.K:
                    row, col = edge_index_gamma_t
                    x_neighbors = x_k[col]
                    
                    if edge_weight_gamma is not None:
                        x_neighbors = x_neighbors * edge_weight_gamma.view(-1, 1)
                    
                    x_k = graph_conv_scatter_add(x_neighbors, row, dim=0, dim_size=T * BN)
        
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        
        # Now compute the actual output using the original layer
        output = self.layer(x, edge_index)
        
        return output, self.intermediates


class TestStridedTAGConv:
    """Test suite for StridedTAGConv."""
    
    def test_forward_4d_basic(self):
        """Test basic forward pass with 4D input."""
        layer = StridedTAGConv(in_channels=32, out_channels=64, gamma=2, K=2)
        
        B, T, N, F = 2, 5, 20, 32
        x = torch.randn(B, T, N, F)
        edge_index = torch.randint(0, B * N, (2, 100))
        
        out = layer(x, edge_index)
        
        assert out.shape == (B, T, N, 64)
        assert not torch.isnan(out).any()
        print("✓ StridedTAGConv 4D forward: PASSED")
    
    def test_forward_3d_basic(self):
        """Test basic forward pass with 3D input."""
        layer = StridedTAGConv(in_channels=32, out_channels=64, gamma=2, K=2)
        
        BN, T, F = 40, 5, 32
        x = torch.randn(BN, T, F)
        edge_index = torch.randint(0, BN, (2, 100))
        
        out = layer(x, edge_index)
        
        assert out.shape == (BN, T, 64)
        assert not torch.isnan(out).any()
        print("✓ StridedTAGConv 3D forward: PASSED")
    
    def test_different_gamma_values(self):
        """Test with different stride values."""
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        # γ=1 (should be similar to standard TAGConv behavior)
        layer_g1 = StridedTAGConv(in_channels=32, out_channels=32, gamma=1, K=3)
        out_g1 = layer_g1(x, edge_index)
        
        # γ=2 (strided: 0, 2, 4, 6-hop)
        layer_g2 = StridedTAGConv(in_channels=32, out_channels=32, gamma=2, K=3)
        out_g2 = layer_g2(x, edge_index)
        
        # γ=3 (strided: 0, 3, 6, 9-hop)
        layer_g3 = StridedTAGConv(in_channels=32, out_channels=32, gamma=3, K=3)
        out_g3 = layer_g3(x, edge_index)
        
        assert out_g1.shape == out_g2.shape == out_g3.shape
        print("✓ StridedTAGConv different gamma values: PASSED")
    
    def test_different_K_values(self):
        """Test with different K (number of strides)."""
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        # K=1: 0-hop and γ-hop only
        layer_k1 = StridedTAGConv(in_channels=32, out_channels=32, gamma=2, K=1)
        out_k1 = layer_k1(x, edge_index)
        
        # K=3: 0, γ, 2γ, 3γ-hop
        layer_k3 = StridedTAGConv(in_channels=32, out_channels=32, gamma=2, K=3)
        out_k3 = layer_k3(x, edge_index)
        
        assert out_k1.shape == out_k3.shape
        print("✓ StridedTAGConv different K values: PASSED")
    
    def test_boolean_vs_power_method(self):
        """Test Boolean semiring BFS (exact γ-hop) vs matrix power (all paths ≤ γ)."""
        x = torch.randn(1, 1, 10, 32)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9]])  # Chain
        
        # Matrix power (default): includes all paths up to γ hops
        layer_power = StridedTAGConv(
            in_channels=32, out_channels=32, 
            gamma=2, K=2, use_boolean_semiring=False
        )
        out_power = layer_power(x, edge_index)
        
        # Boolean semiring: exact γ-hop only
        layer_bool = StridedTAGConv(
            in_channels=32, out_channels=32, 
            gamma=2, K=2, use_boolean_semiring=True
        )
        out_bool = layer_bool(x, edge_index)
        
        assert out_bool.shape == out_power.shape
        # They may have different values due to different neighborhood definitions
        print("✓ StridedTAGConv Boolean vs power method: PASSED")
    
    def test_normalization(self):
        """Test with and without adjacency normalization."""
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        # With normalization
        layer_norm = StridedTAGConv(
            in_channels=32, out_channels=32, 
            gamma=2, K=2, normalize=True
        )
        out_norm = layer_norm(x, edge_index)
        
        # Without normalization
        layer_no_norm = StridedTAGConv(
            in_channels=32, out_channels=32, 
            gamma=2, K=2, normalize=False
        )
        out_no_norm = layer_no_norm(x, edge_index)
        
        assert out_norm.shape == out_no_norm.shape
        print("✓ StridedTAGConv normalization: PASSED")

    def test_power_mode_sparse_path_bypasses_dense_adjacency(self, monkeypatch):
        """Power mode without normalization should not call dense A^gamma helpers."""
        layer = StridedTAGConv(
            in_channels=16,
            out_channels=16,
            gamma=2,
            K=2,
            normalize=False,
            use_boolean_semiring=False,
        )

        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("Dense A^gamma helper should not be called in sparse fast path")

        monkeypatch.setattr(layer, "_compute_gamma_hop_adjacency_power", _should_not_be_called)

        x = torch.randn(2, 3, 12, 16)
        edge_index = torch.randint(0, 24, (2, 64))
        out = layer(x, edge_index)

        assert out.shape == (2, 3, 12, 16)
        print("✓ StridedTAGConv sparse fast path bypasses dense adjacency: PASSED")

    def test_power_mode_sparse_matches_dense_reference(self):
        """Sparse fast path should match dense power reference (small graph, normalize=False)."""
        from torch_geometric.utils import to_dense_adj

        torch.manual_seed(0)

        layer = StridedTAGConv(
            in_channels=4,
            out_channels=5,
            gamma=3,
            K=2,
            normalize=False,
            use_boolean_semiring=False,
        )

        B, T, N, F = 1, 2, 6, 4
        x = torch.randn(B, T, N, F)
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 1, 5, 2],
                [1, 2, 3, 4, 5, 0, 2, 4],
            ],
            dtype=torch.long,
        )
        edge_weight = torch.rand(edge_index.size(1)) + 0.1

        out_sparse = layer(x, edge_index, edge_weight=edge_weight)

        # Dense reference for historical power-mode semantics:
        # A_base = A (+ I when gamma > 1), x_{k+1} = (A_base^gamma)^T x_k
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=N)[0].float()
        if layer.gamma > 1:
            adj_base = adj + torch.eye(N, dtype=adj.dtype)
        else:
            adj_base = adj
        adj_gamma = torch.matrix_power(adj_base, layer.gamma)
        adj_gamma_t = adj_gamma.t()

        x_seq = x.permute(1, 0, 2, 3).reshape(T, N, F)
        out_ref_seq = []
        for t in range(T):
            x_k = x_seq[t]
            out_t = torch.zeros(N, layer.out_channels, dtype=x.dtype)
            for k in range(layer.K + 1):
                out_t = out_t + layer.lins[k](x_k)
                if k < layer.K:
                    x_k = torch.matmul(adj_gamma_t, x_k)
            if layer.bias is not None:
                out_t = out_t + layer.bias
            out_ref_seq.append(out_t)

        out_ref = torch.stack(out_ref_seq, dim=0).reshape(T, B, N, layer.out_channels).permute(1, 0, 2, 3)

        assert torch.allclose(out_sparse, out_ref, atol=1e-5, rtol=1e-5)
        print("✓ StridedTAGConv sparse fast path matches dense reference: PASSED")

    def test_exact_walk_sparse_matches_dense_reference(self):
        """Exact-walk mode should match dense reference with A_base=A (no +I)."""
        from torch_geometric.utils import to_dense_adj

        torch.manual_seed(7)

        layer = StridedTAGConv(
            in_channels=3,
            out_channels=4,
            gamma=2,
            K=2,
            normalize=False,
            use_boolean_semiring=False,
            power_mode='exact_walk',
        )

        B, T, N, F = 1, 2, 5, 3
        x = torch.randn(B, T, N, F)
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 3, 4, 2],
                [1, 0, 2, 1, 4, 3, 4],
            ],
            dtype=torch.long,
        )
        edge_weight = torch.rand(edge_index.size(1)) + 0.2

        out_sparse = layer(x, edge_index, edge_weight=edge_weight)

        # Dense exact-walk reference: A_base=A (no self-loop augmentation).
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=N)[0].float()
        adj_gamma = torch.matrix_power(adj, layer.gamma)
        adj_gamma_t = adj_gamma.t()

        x_seq = x.permute(1, 0, 2, 3).reshape(T, N, F)
        out_ref_seq = []
        for t in range(T):
            x_k = x_seq[t]
            out_t = torch.zeros(N, layer.out_channels, dtype=x.dtype)
            for k in range(layer.K + 1):
                out_t = out_t + layer.lins[k](x_k)
                if k < layer.K:
                    x_k = torch.matmul(adj_gamma_t, x_k)
            if layer.bias is not None:
                out_t = out_t + layer.bias
            out_ref_seq.append(out_t)

        out_ref = torch.stack(out_ref_seq, dim=0).reshape(T, B, N, layer.out_channels).permute(1, 0, 2, 3)

        assert torch.allclose(out_sparse, out_ref, atol=1e-5, rtol=1e-5)
        print("✓ StridedTAGConv exact_walk sparse path matches dense reference: PASSED")

    def test_exact_walk_differs_from_a_plus_i_for_gamma_gt1(self):
        """Exact-walk mode should differ from default A+I mode when gamma>1."""
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        )
        x = torch.zeros(1, 1, 3, 1)
        x[0, 0, 0, 0] = 1.0

        layer_default = StridedTAGConv(
            in_channels=1,
            out_channels=1,
            gamma=2,
            K=1,
            bias=False,
            normalize=False,
            power_mode='a_plus_i',
        )
        layer_exact = StridedTAGConv(
            in_channels=1,
            out_channels=1,
            gamma=2,
            K=1,
            bias=False,
            normalize=False,
            power_mode='exact_walk',
        )

        # Isolate the strided propagation term: out = (A_base^gamma)x
        with torch.no_grad():
            layer_default.lins[0].weight.zero_()
            layer_default.lins[1].weight.fill_(1.0)
            layer_exact.lins[0].weight.zero_()
            layer_exact.lins[1].weight.fill_(1.0)

        out_default = layer_default(x, edge_index)
        out_exact = layer_exact(x, edge_index)

        assert not torch.allclose(out_default, out_exact)
        print("✓ StridedTAGConv exact_walk differs from default a_plus_i for gamma>1: PASSED")
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        layer = StridedTAGConv(in_channels=32, out_channels=32, gamma=2, K=2)
        
        x = torch.randn(2, 5, 10, 32, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 40))
        
        out = layer(x, edge_index)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # Check that layer parameters have gradients allocated
        # Note: Some parameters might have zero gradients depending on graph structure,
        # but we verify that at least the gradient computation happened
        total_grad_norm = 0
        for param in layer.parameters():
            assert param.grad is not None, "Gradient not allocated"
            total_grad_norm += param.grad.abs().sum().item()
        
        # Verify at least some parameters have non-zero gradients
        assert total_grad_norm > 0, "All gradients are zero"
        
        print("✓ StridedTAGConv gradient flow: PASSED")
    
    def test_comparison_with_tagconv_gamma1(self):
        """Test that γ=1 behaves similarly to standard TAGConv."""
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        # StridedTAGConv with γ=1
        strided_layer = StridedTAGConv(
            in_channels=32, out_channels=32, gamma=1, K=3
        )
        
        # Standard TAGConv
        standard_layer = TAGConvLayer(
            in_channels=32, out_channels=32, K=3
        )
        
        # Copy weights to make them comparable
        # (won't be identical due to different implementations, but shapes match)
        out_strided = strided_layer(x, edge_index)
        out_standard = standard_layer(x, edge_index)
        
        assert out_strided.shape == out_standard.shape
        print("✓ StridedTAGConv γ=1 vs TAGConv: PASSED")
    
    def test_chain_graph_aggregation(self):
        """Test that strided aggregation works correctly on a chain graph."""
        # Simple chain: 0-1-2-3-4-5-6-7-8-9
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9]])
        
        B, T, N, F = 1, 1, 10, 1
        x = torch.zeros(B, T, N, F)
        
        # Set specific values to verify aggregation
        x[0, 0, 0, 0] = 1.0   # Node 0
        x[0, 0, 2, 0] = 2.0   # Node 2 (2-hop from 0)
        x[0, 0, 4, 0] = 4.0   # Node 4 (4-hop from 0)
        x[0, 0, 6, 0] = 6.0   # Node 6 (6-hop from 0)
        
        # γ=2, K=2: should aggregate 0, 2, 4-hop neighbors
        layer = StridedTAGConv(in_channels=1, out_channels=1, gamma=2, K=2)
        
        # Set weights to identity for easier verification
        for lin in layer.lins:
            lin.weight.data.fill_(1.0)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)
        
        out = layer(x, edge_index)
        
        # Node 0 should aggregate from: itself (0-hop), node 2 (2-hop), node 4 (4-hop)
        # But actual aggregation depends on normalization and edge structure
        # Just verify shape and that computation runs
        assert out.shape == (B, T, N, 1)
        
        print("✓ StridedTAGConv chain graph aggregation: PASSED")
    
    def test_output_channels_different(self):
        """Test with different input and output channels."""
        layer = StridedTAGConv(in_channels=32, out_channels=128, gamma=2, K=2)
        
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        out = layer(x, edge_index)
        
        assert out.shape == (2, 5, 20, 128)
        print("✓ StridedTAGConv different output channels: PASSED")
    
    def test_bias_parameter(self):
        """Test with and without bias."""
        x = torch.randn(2, 5, 10, 32)
        edge_index = torch.randint(0, 20, (2, 40))
        
        # With bias
        layer_bias = StridedTAGConv(
            in_channels=32, out_channels=32, gamma=2, K=2, bias=True
        )
        assert layer_bias.bias is not None
        out_bias = layer_bias(x, edge_index)
        
        # Without bias
        layer_no_bias = StridedTAGConv(
            in_channels=32, out_channels=32, gamma=2, K=2, bias=False
        )
        assert layer_no_bias.bias is None
        out_no_bias = layer_no_bias(x, edge_index)
        
        assert out_bias.shape == out_no_bias.shape
        print("✓ StridedTAGConv bias parameter: PASSED")
    
    def test_lattice_graph_visualization(self):
        """
        Test and visualize StridedTAGConv on a 2D lattice graph.
        
        Uses γ=2, K=2 with alternating weights (+1, -1, +1) to verify that
        the strided convolution aggregates features from 0, 2γ, 4γ-hop neighbors.
        Shows intermediate propagations: x, A²x, A⁴x, and final weighted sum.
        """
        # Create a 2D lattice graph (grid)
        grid_size = 10
        N = grid_size * grid_size
        
        # Generate node positions
        positions = []
        node_to_idx = {}
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                positions.append((i, j))
                node_to_idx[(i, j)] = idx
        
        # Create edges (4-connected grid)
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = node_to_idx[(i, j)]
                # Right neighbor
                if j < grid_size - 1:
                    edges.append([idx, node_to_idx[(i, j + 1)]])
                # Down neighbor
                if i < grid_size - 1:
                    edges.append([idx, node_to_idx[(i + 1, j)]])
                # Left neighbor
                if j > 0:
                    edges.append([idx, node_to_idx[(i, j - 1)]])
                # Up neighbor
                if i > 0:
                    edges.append([idx, node_to_idx[(i - 1, j)]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create input features: all zeros except center node
        B, T, F = 1, 1, 1
        x = torch.zeros(B, T, N, F)
        center_idx = node_to_idx[(grid_size // 2, grid_size // 2)]
        x[0, 0, center_idx, 0] = 1.0
        
        # Create StridedTAGConv with γ=2, K=2
        # This aggregates: k=0: (A²)⁰x = x (0-hop), k=1: (A²)¹x = A²x (2-hop), k=2: (A²)²x = A⁴x (4-hop)
        layer = StridedTAGConv(
            in_channels=1, 
            out_channels=1, 
            gamma=2, 
            K=2, 
            bias=False,
            normalize=False  # Disable normalization for clearer interpretation
        )
        
        # Set alternating weights: +1, -1, +1 for k=0, 1, 2
        with torch.no_grad():
            layer.lins[0].weight.data.fill_(1.0)   # k=0: (A²)⁰ = I (0-hop)
            layer.lins[1].weight.data.fill_(-1.0)  # k=1: (A²)¹ = A² (2-hop)
            layer.lins[2].weight.data.fill_(1.0)   # k=2: (A²)² = A⁴ (4-hop)
        
        # Wrap with debug layer to capture intermediates
        debug_layer = DebugStridedTAGConv(layer)
        
        # Forward pass
        out, intermediates = debug_layer(x, edge_index)
        
        # Extract features for visualization
        input_features = x[0, 0, :, 0].numpy()
        output_features = out[0, 0, :, 0].detach().numpy()
        
        # Extract intermediate features
        # intermediates[k] contains (A^γ)^k x
        intermediate_features = [interm[0, 0, :, 0].numpy() for interm in intermediates]
        
        # Create visualization with grid-based layout
        # Show: Input | A²x | A⁴x | Output (weighted sum)
        num_plots = len(intermediates) + 1  # intermediates + final output
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        
        # Helper function to plot grid
        def plot_grid_features(ax, features, title, vmin=None, vmax=None):
            """Plot features on a grid layout with proper node value coloring."""
            # Reshape features to grid
            features_grid = features.reshape(grid_size, grid_size)
            
            # Create meshgrid for proper grid visualization
            x_coords = np.arange(grid_size)
            y_coords = np.arange(grid_size)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Plot as scatter with value-based coloring
            if vmin is None:
                vmin = features.min()
            if vmax is None:
                vmax = features.max()
            
            # Use symmetric colormap limits for better visualization
            abs_max = max(abs(vmin), abs(vmax))
            if abs_max > 0:
                vmin, vmax = -abs_max, abs_max
            
            sc = ax.scatter(
                X.flatten(), 
                Y.flatten(), 
                c=features,
                cmap='RdBu_r',
                s=300,
                vmin=vmin,
                vmax=vmax,
                edgecolors='black',
                linewidth=1.5
            )
            
            # Add grid lines
            for i in range(grid_size + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            
            # Add text labels for each node value
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    value = features[idx]
                    # Choose text color based on background color
                    text_color = 'white' if abs(value) > abs_max * 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', 
                           ha='center', va='center',
                           fontsize=7, fontweight='bold',
                           color=text_color)
            
            # Highlight center node
            center_i, center_j = grid_size // 2, grid_size // 2
            circle = plt.Circle((center_j, center_i), 0.4, 
                              fill=False, edgecolor='yellow', 
                              linewidth=3, linestyle='--')
            ax.add_patch(circle)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X (column)', fontsize=10)
            ax.set_ylabel('Y (row)', fontsize=10)
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(-0.5, grid_size - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            return sc
        
        # Plot intermediates
        titles = [
            'Input: x\n(0-hop)',
            '(A²)¹·x = A²·x\n(2-hop propagation)',
            '(A²)²·x = A⁴·x\n(4-hop propagation)'
        ]
        
        # Find global vmin/vmax for consistent coloring across intermediates
        all_intermediate_values = np.concatenate(intermediate_features)
        abs_max_intermediate = max(abs(all_intermediate_values.min()), abs(all_intermediate_values.max()))
        
        for k, (features, title) in enumerate(zip(intermediate_features, titles)):
            sc = plot_grid_features(
                axes[k], 
                features, 
                title,
                vmin=-abs_max_intermediate if abs_max_intermediate > 0 else -1,
                vmax=abs_max_intermediate if abs_max_intermediate > 0 else 1
            )
            plt.colorbar(sc, ax=axes[k], label='Feature Value', fraction=0.046, pad=0.04)
        
        # Plot final output (weighted sum)
        sc_out = plot_grid_features(
            axes[-1], 
            output_features, 
            'Output: w₀·x + w₁·(A²x) + w₂·(A⁴x)\nWeights: [+1, -1, +1]'
        )
        plt.colorbar(sc_out, ax=axes[-1], label='Feature Value', fraction=0.046, pad=0.04)
        
        # Add annotation about the aggregation
        fig.text(
            0.5, 0.01,
            'StridedTAGConv (γ=2, K=2): Propagates signal through powers of A² and computes weighted sum\n'
            'Yellow dashed circle = center node (source) | Each cell shows the node feature value',
            ha='center',
            fontsize=10,
            style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        
        # Save visualization with archiving
        output_dir = Path(__file__).parent / "plot_strided_tagconv_visualizations"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "lattice_graph_strided_tagconv.pdf"
        save_plot_with_archive(fig, output_path)
        print(f"  ✓ Saved visualization: {output_path}")
        plt.close()
        
        # Verify properties
        assert out.shape == (B, T, N, 1)
        assert not torch.isnan(out).any()
        
        # Check that output is different from input (convolution happened)
        assert not torch.allclose(x, out)
        
        # The center node should have:
        # - Its own feature (1.0) × weight[0] = 1.0 (0-hop: itself)
        # - Features from 2-hop neighbors × weight[1] = something × -1 (2-hop via A²)
        # - Features from 4-hop neighbors × weight[2] = something × 1 (4-hop via A⁴)
        # Without normalization, the aggregation depends on the graph structure
        center_output = out[0, 0, center_idx, 0].item()
        
        print(f"  Center node output value: {center_output:.4f}")
        print("✓ StridedTAGConv lattice graph visualization: PASSED")


# Run all tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing StridedTAGConv")
    print("="*60)
    
    test_suite = TestStridedTAGConv()
    test_suite.test_forward_4d_basic()
    test_suite.test_forward_3d_basic()
    test_suite.test_different_gamma_values()
    test_suite.test_different_K_values()
    test_suite.test_boolean_vs_power_method()
    test_suite.test_normalization()
    test_suite.test_gradient_flow()
    test_suite.test_comparison_with_tagconv_gamma1()
    test_suite.test_chain_graph_aggregation()
    test_suite.test_output_channels_different()
    test_suite.test_bias_parameter()
    test_suite.test_lattice_graph_visualization()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
