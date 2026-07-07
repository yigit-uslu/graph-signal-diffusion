"""
Power Allocation GNN for Wireless Resource Allocation.

This module implements a GNN-based primal policy for learning power allocations
in wireless interference networks. The GNN processes a graph representation of
the network (nodes = receivers, edges = interference relationships) and outputs
power allocations for transmitters.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TAGConv
from typing import Optional, Literal


class PowerAllocationGNN(nn.Module):
    """
    GNN-based power allocation policy for wireless networks.
    
    Architecture:
        - Input projection: (n, 2) → (n, hidden_dim)
        - L residual GNN blocks with TAGConv (K-hop aggregation)
        - Output MLP: (n, hidden_dim) → (n, 1) → sigmoid × P_max
        - Scatter-add to map receiver embeddings → transmitter powers
    
    The GNN processes graphs where:
    - Nodes represent receivers
    - Self-loops weighted by direct channel gain
    - Edges weighted by interference channel gains
    - Node features: [direct_signal_strength, total_interference_potential]
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension (default: 2)
    hidden_dim : int
        Hidden dimension for GNN layers (default: 64)
    num_layers : int
        Number of ResidualGNNBlock layers (default: 3)
    K : int
        Number of hops for TAGConv aggregation (default: 2)
    P_max : float
        Maximum transmit power in linear scale (default: 0.01 = 10 mW)
    norm_type : str
        Normalization type: 'layer', 'group', 'instance', 'graph' (default: 'group')
    num_groups : int
        Number of groups for GroupNorm (default: 8)
    dropout : float
        Dropout rate (default: 0.1)
    activation : str
        Activation function: 'relu', 'silu', 'gelu', 'leaky_relu' (default: 'silu')
    
    Examples
    --------
    >>> model = PowerAllocationGNN(input_dim=2, hidden_dim=64, num_layers=3, K=2)
    >>> # Single graph with 10 receivers
    >>> x = torch.randn(10, 2)
    >>> edge_index = torch.randint(0, 10, (2, 50))
    >>> edge_weight = torch.rand(50)
    >>> associations = torch.eye(10)  # One-to-one pairing
    >>> power = model(x, edge_index, edge_weight, associations)
    >>> print(power.shape)  # (10,) - one power value per transmitter
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        K: int = 2,
        P_max: float = 0.01,
        norm_type: Literal['layer', 'group', 'instance', 'graph'] = 'group',
        num_groups: int = 8,
        dropout: float = 0.1,
        activation: Literal['relu', 'silu', 'gelu', 'leaky_relu'] = 'silu',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = K
        self.P_max = P_max
        
        # Input projection: (n, input_dim) → (n, hidden_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Stack of L simple GNN layers with TAGConv
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                nn.ModuleList([
                    TAGConv(hidden_dim, hidden_dim, K=K),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ])
            )
        
        # Output MLP: (n, hidden_dim) → (n, 1)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights for numerical stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values to prevent NaN in early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with small gain
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        associations: torch.Tensor = None,  # Not used; power-to-receiver mapping done by caller
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: batched graph → receiver embeddings → power per receiver.

        PyG automatically batches graphs by concatenating:
        - x: (total_nodes, input_dim) - all nodes from all graphs
        - edge_index: (2, total_edges) - with proper node offset per graph
        - edge_weight: (total_edges,) - all edge weights concatenated
        - batch: (total_nodes,) - batch[i] = graph_id for node i

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (total_nodes, input_dim)
        edge_index : torch.Tensor
            Edge connectivity, shape (2, total_edges)
        edge_weight : torch.Tensor
            Edge weights, shape (total_edges,)
        associations : torch.Tensor, optional
            Not used; transmitter-to-receiver power mapping is handled by the caller
            via receiver_to_transmitter_power() after the forward pass.
        batch : torch.Tensor, optional
            Batch assignment, shape (total_nodes,). Accepted for API consistency and
            forward-compatibility, but not consumed by any layer in the current
            architecture (LayerNorm + TAGConv are both node-local operations).

        Returns
        -------
        power : torch.Tensor
            Power per receiver, shape (total_nodes,)
            Concatenated across all graphs in batch
        """
        # Input projection
        h = self.input_projection(x)  # (total_nodes, hidden_dim)
        
        # Pass through GNN layers - TAGConv handles batched graphs automatically
        for i, (conv, norm, act, dropout) in enumerate(self.gnn_layers):
            h_new = conv(h, edge_index, edge_weight)
            h_new = norm(h_new)
            h_new = act(h_new)
            h_new = dropout(h_new)
            h = h + h_new  # Residual
        
        # Output MLP
        h_out = self.output_mlp(h)  # (total_nodes, 1)
        h_out = h_out.squeeze(-1)  # (total_nodes,)
        
        # Sigmoid and scale
        power_per_receiver = torch.sigmoid(h_out) * self.P_max
        
        return power_per_receiver
    
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, K={self.K}, P_max={self.P_max}"
        )
