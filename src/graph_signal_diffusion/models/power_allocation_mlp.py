"""
Simple MLP-based Power Allocation for debugging.

This module implements an MLP baseline that processes each receiver independently
without using graph structure. Useful for debugging NaN issues.
"""

import torch
import torch.nn as nn
from typing import Optional


class PowerAllocationMLP(nn.Module):
    """
    Simple MLP-based power allocation (no graph structure).
    
    Architecture:
        - Input: (n, input_dim) node features
        - MLP: input_dim → hidden_dim → hidden_dim → 1
        - Output: sigmoid × P_max per receiver
        - Aggregate using associations matrix
    
    Parameters
    ----------
    input_dim : int
        Input node feature dimension (default: 2)
    hidden_dim : int
        Hidden dimension for MLP (default: 64)
    num_layers : int
        Number of hidden layers (default: 2)
    P_max : float
        Maximum transmit power in linear scale (default: 0.01 = 10 mW)
    dropout : float
        Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        P_max: float = 0.01,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.P_max = P_max
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
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
        edge_index: torch.Tensor,  # Not used, for API compatibility
        edge_weight: torch.Tensor,  # Not used, for API compatibility
        associations: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: node features → MLP → transmitter powers.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (n, input_dim)
        edge_index : torch.Tensor
            Not used (for API compatibility with GNN)
        edge_weight : torch.Tensor
            Not used (for API compatibility with GNN)
        associations : torch.Tensor
            TX-RX association matrix, shape (m, n)
            associations[i, j] = 1 if TX_i paired with RX_j
        batch : torch.Tensor, optional
            Not used (for API compatibility with GNN)
        
        Returns
        -------
        power : torch.Tensor
            Power allocations, shape (m,)
        """
        # DEBUG: Check input for NaN
        if torch.isnan(x).any():
            print(f"⚠️ MLP: NaN in input x!")
        
        # Pass through MLP: (n, input_dim) → (n, 1)
        h = self.mlp(x)  # (n, 1)
        
        # DEBUG: Check after MLP
        if torch.isnan(h).any():
            print(f"⚠️ MLP: NaN after mlp!")
            print(f"  h stats: min={h.min().item():.4f}, max={h.max().item():.4f}")
            print(f"  x stats: min={x.min().item():.4f}, max={x.max().item():.4f}")
            # Check MLP weights
            for i, layer in enumerate(self.mlp):
                if isinstance(layer, nn.Linear):
                    if torch.isnan(layer.weight).any():
                        print(f"  NaN in mlp layer {i} weight")
                    if layer.bias is not None and torch.isnan(layer.bias).any():
                        print(f"  NaN in mlp layer {i} bias")
        
        h = h.squeeze(-1)  # (n,)
        
        # Apply sigmoid to get [0, 1] range, then scale to [0, P_max]
        power_per_receiver = torch.sigmoid(h) * self.P_max  # (n,)
        
        # DEBUG: Check after sigmoid
        if torch.isnan(power_per_receiver).any():
            print(f"⚠️ MLP: NaN after sigmoid!")
            print(f"  power_per_receiver stats: min={power_per_receiver.min().item():.6e}, max={power_per_receiver.max().item():.6e}")
        
        # Map receiver powers to transmitter powers using associations
        if batch is None:
            # Single graph case
            m, n = associations.shape
            assert n == x.shape[0], f"Associations has {n} receivers but x has {x.shape[0]} nodes"
            
            # power[i] = sum_j associations[i, j] * power_per_receiver[j]
            power = associations @ power_per_receiver  # (m,)
        else:
            raise NotImplementedError("Batched graph processing not yet implemented.")
        
        return power
    
    def forward_batch(
        self,
        batch_data,
        associations_list: list,
    ) -> list:
        """
        Process a batch of graphs and return powers for each graph.
        
        Parameters
        ----------
        batch_data : Batch
            PyTorch Geometric Batch object containing multiple graphs
        associations_list : list
            List of association matrices, one per graph
        
        Returns
        -------
        powers : list
            List of power allocations, one per graph
        """
        # Separate batch into individual graphs
        data_list = batch_data.to_data_list()
        
        powers = []
        for i, (data, associations) in enumerate(zip(data_list, associations_list)):
            # Process each graph independently
            power = self.forward(
                x=data.x,
                edge_index=data.edge_index,
                edge_weight=data.edge_weight,
                associations=associations,
                batch=None,
            )
            powers.append(power)
        
        return powers
