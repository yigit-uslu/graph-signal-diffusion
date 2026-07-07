import torch
import torch.nn as nn
from torch_geometric.data import Data

from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from typing import Optional

# from graph_signal_diffusion.models.registry import MODEL_REGISTRY

# @MODEL_REGISTRY.register("unet")
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        obs_channels: int,
        channels: int,
        channel_multipliers: list,
        in_timesteps: int,
        obs_timesteps: int,
        out_timesteps: int,
        num_res_blocks: int,
        attention_resolutions: list,
        dropout: float,
        time_embed_dim: int,
        num_classes: int = None,
        **kwargs,  # absorb non-init config fields like `name`
    ):
        super().__init__()
        self.debug_print = lambda msg: None  # Placeholder, will be injected by trainer

        self.in_channels = in_channels
        self.obs_channels = obs_channels
        self.out_channels = out_channels
        self.channels = channels
        self.channel_multipliers = channel_multipliers
        self.in_timesteps = in_timesteps 
        self.obs_timesteps = obs_timesteps
        self.out_timesteps = out_timesteps
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.time_embed_dim = time_embed_dim
        self.num_classes = num_classes

        # TODO: Implement your UNet architecture here
        # Time embedding, down blocks, middle block, up blocks, output projection
        # self.mlp = nn.Sequential(
        #     nn.Linear(time_embed_dim, time_embed_dim * 4),
        #     nn.GELU(),
        #     nn.Linear(time_embed_dim * 4, time_embed_dim),
        # )

        self.x_embedding = nn.Sequential(
            nn.Linear(obs_channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

        self.y_embedding = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

        self.mlp_timesteps = nn.Sequential(
            nn.Linear(in_timesteps + obs_timesteps + 1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, out_timesteps)
        )

        self.mlp_channels = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, out_channels)
        )
        # pass
    
    @jaxtyped(typechecker=beartype)
    def forward(
            self,
            y: Float[torch.Tensor, "batch timesteps nodes features"],
            t: Int[torch.Tensor, "batch"],
            data: Optional[Data] = None
        ) -> Float[torch.Tensor, "batch timesteps nodes features"]:

        B, T, N, F = y.shape
        y = self.y_embedding(y)  # ..., F -> ..., H
        # y = y.reshape(-1, F)

        if data and data.x is not None:
            # Reshape and permute observations to [B, T_x, N, F_x]
            x = data.x.view(B, N, *data.x.shape[1:]).permute(0, 2, 1, 3)  # [B, T_x, N, F_x]
            x = self.x_embedding(x) # ..., F_x -> ..., H

            z = torch.cat([y, x], dim=1)  # Concatenate along time dimension [B, T + T_x, N, H]
            t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, N, z.size(-1)).float()
            z = torch.cat([z, t], dim = 1) # Simple concatenation of time embedding [B, T + T_x + 1, N, H] 

            # MLP on time dimension 
            z = self.mlp_timesteps(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, T_out, N, H] 
            pred = self.mlp_channels(z) # [B, T_out, N, F_out]
        
        else:
            z = y  # No observations provided [B, T, N, H]
            t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, N, z.size(-1)).float()
            z = torch.cat([z, t], dim = 1) # Simple concatenation of time embedding [B, T + 1, N, H]   

            pred = z   # Placeholder for actual model logic        

        # z = y + t.float()  # Simple operation for demonstration

        # w = self.mlp(z) # Simple MLP for demonstration
        # pred = w.reshape(B, T, N, F)
 
        return pred  # Placeholder return, replace with actual forward pass logic