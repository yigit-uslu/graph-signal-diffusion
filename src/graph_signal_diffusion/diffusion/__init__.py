from graph_signal_diffusion.diffusion.base import BaseDiffusion
from graph_signal_diffusion.diffusion.ddpm import DDPM
from graph_signal_diffusion.diffusion.revin_resolvers import register_revin_resolvers

# Register ${revin_alpha:w,b} so diffusion configs can interpolate
# `revin_sigma_correction` from `revin_blend_weight` automatically.
register_revin_resolvers()

__all__ = ["BaseDiffusion", "DDPM", "register_revin_resolvers"]