import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union
import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import Data

from jaxtyping import Float, Int, jaxtyped
from beartype import beartype

from graph_signal_diffusion.diffusion.ddpm import DDPM, _SamplingProbeState


class DDIM(DDPM):
    """
    Denoising Diffusion Implicit Model (Song et al., 2021).
    
    DDIM accelerates sampling by:
    1. Using a subsequence of timesteps (e.g., every 10th step)
    2. Employing a deterministic (or partially stochastic) reverse process
    3. Allowing direct jumps between non-consecutive timesteps
    
    When eta=0: fully deterministic sampling (fastest)
    When eta=1: equivalent to DDPM sampling
    """
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        loss_type: str = "l2",
        parameterization: str = "eps",
        clip_denoised: bool = True,
        cosine_s: float = 0.008,
        sampling_timesteps: Optional[int] = None,
        ddim_eta: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            model: Denoising model
            num_timesteps: Total timesteps in diffusion process (training)
            sampling_timesteps: Number of timesteps for sampling (inference). 
                               If None, uses num_timesteps (equivalent to DDPM)
            ddim_eta: Stochasticity parameter. 0 = deterministic, 1 = DDPM-like
            Other args: same as DDPM
        """
        super().__init__(
            model=model,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            loss_type=loss_type,
            parameterization=parameterization,
            clip_denoised=clip_denoised,
            cosine_s=cosine_s,
            **kwargs,
        )
        
        self.sampling_timesteps = int(sampling_timesteps or num_timesteps)
        if self.sampling_timesteps <= 0:
            raise ValueError(f"sampling_timesteps must be positive, got {self.sampling_timesteps}")
        self.ddim_eta = ddim_eta
        
        # Create sampling schedule (subsequence of training timesteps)
        self.ddim_timesteps = self._make_ddim_timesteps()
        
    def _make_ddim_timesteps(self) -> torch.Tensor:
        """
        Create a subsequence of timesteps for accelerated sampling.
        Uses linspace for uniform spacing including endpoints.
        
        Returns timesteps in DESCENDING order (T-1 to 0) for sampling.
        Note: Does NOT include 0 - the final step to 0 is handled specially.
        """
        if self.sampling_timesteps >= self.num_timesteps:
            # No acceleration, use all timesteps (excluding 0, handled in sample loop)
            return torch.arange(self.num_timesteps - 1, 0, -1)
        
        # Build a strictly descending, duplicate-free schedule in [1, T-1].
        # Using float linspace + floor avoids collisions caused by integer linspace rounding.
        idx = torch.linspace(0, self.num_timesteps - 2, self.sampling_timesteps, dtype=torch.float64)
        ddim_timesteps = (self.num_timesteps - 1) - torch.floor(idx).to(torch.long)
        return ddim_timesteps
    
    @torch.inference_mode()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        data: Optional[Data] = None,
        eta: Optional[float] = None,
        use_amp: bool = False,
        return_selector_sampling_diagnostics: bool = False,
        selector_probe_timesteps: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        """Generate samples using DDIM accelerated sampling.

        Args:
            shape: (batch, timesteps, nodes, features)
            device: Device to run on
            data: Graph data for conditional generation
            eta: Override default ddim_eta for this sampling run
            use_amp: Use automatic mixed precision (FP16) for faster inference
            return_selector_sampling_diagnostics: If True, also return a compact
                selector diagnostics dictionary aggregated across probe timesteps.
            selector_probe_timesteps: Optional explicit probe timesteps in DDPM index
                space [0, T-1]. DDIM probes nearest visited DDIM steps.

        Returns:
            Generated samples, and optionally sampling diagnostics.
        """
        eta = float(eta if eta is not None else self.ddim_eta)
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta must be in [0, 1], got {eta}")

        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
        probe_ctx = self._sampling_probe_context() if return_selector_sampling_diagnostics else nullcontext()
        probe_state: Optional[_SamplingProbeState] = None

        B, N = shape[0], shape[2]

        with probe_ctx:
            with autocast_ctx:
                x_t = torch.randn(shape, device=device)
                timesteps = self.ddim_timesteps.to(device)
                timesteps_list = [int(t.item()) for t in timesteps]

                if return_selector_sampling_diagnostics:
                    # Resolve probe timesteps, mapping requested ones to nearest DDIM step.
                    requested = self._resolve_probe_timesteps(selector_probe_timesteps)
                    if timesteps_list:
                        resolved: List[int] = []
                        seen: set = set()
                        for t_req in requested:
                            nearest = min(
                                timesteps_list,
                                key=lambda t_seen: (abs(t_seen - int(t_req)), -t_seen),
                            )
                            if nearest not in seen:
                                seen.add(nearest)
                                resolved.append(nearest)
                        probe_timesteps_resolved = sorted(resolved, reverse=True)
                    else:
                        probe_timesteps_resolved = []

                    graph_keys = self._resolve_probe_graph_keys(data, B)
                    probe_state = _SamplingProbeState(
                        probe_timesteps=probe_timesteps_resolved,
                        num_nodes=N,
                        device=device,
                        graph_keys=graph_keys,
                    )

                # Conditioning and graph structure are static across DDIM steps.
                edge_index = data.edge_index if data is not None and hasattr(data, "edge_index") else None
                edge_weight = data.edge_weight if data is not None and hasattr(data, "edge_weight") else None
                u_0 = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if data is not None and hasattr(data, "x") and data.x is not None else None

                # RevIN: compute stats from conditioning for denormalization after sampling
                revin_mu, revin_sigma = None, None
                if self.revin_enabled:
                    if u_0 is None:
                        raise ValueError("RevIN requires conditioning tensor (data.x)")
                    target_idx = self._resolve_revin_target_idx(data)
                    revin_mu, revin_sigma = self._compute_revin_stats(u_0, target_idx)

                # Iterate through timesteps (already in descending order, excluding 0).
                for i, t_curr in enumerate(tqdm.tqdm(timesteps, desc="DDIM Sampling", unit="step")):
                    t_curr_int = int(t_curr.item())
                    t = torch.full((B,), t_curr_int, device=device, dtype=torch.long)

                    # Get next timestep (use timesteps[i+1] or virtual clean endpoint for final step).
                    if i + 1 < len(timesteps):
                        t_next_val = int(timesteps[i + 1].item())
                    else:
                        t_next_val = -1  # Sentinel: alpha_bar_next = 1.0 (clean x0 endpoint)
                    t_next = torch.full((B,), t_next_val, device=device, dtype=torch.long)

                    # Model prediction
                    pred, _ = self.model(
                        x=x_t, timesteps=t, edge_index=edge_index,
                        edge_weight=edge_weight, cond=u_0,
                        return_intermediates=False,
                    )

                    # Convert prediction to x0 and eps
                    x0_pred, eps_pred = self._pred_to_x0_eps(x_t, t, pred)

                    # DDIM update step
                    x_t = self._ddim_step(
                        x_t=x_t, t=t, t_next=t_next,
                        x0_pred=x0_pred, eps_pred=eps_pred, eta=eta,
                    )

                    if probe_state is not None:
                        probe_state.accumulate(t_curr_int, self.model, B)

        # RevIN: denormalize from RevIN space back to global-standardized space
        if self.revin_enabled:
            x_t = self._revin_denormalize(x_t, revin_mu, revin_sigma)

        if probe_state is not None:
            return x_t, probe_state.finalize()
        return x_t


    @jaxtyped(typechecker=beartype)
    def _ddim_step(
        self,
        x_t: Float[torch.Tensor, "batch timesteps nodes features"],
        t: Int[torch.Tensor, "batch"],
        t_next: Int[torch.Tensor, "batch"],
        x0_pred: Float[torch.Tensor, "batch timesteps nodes features"],
        eps_pred: Float[torch.Tensor, "batch timesteps nodes features"],
        eta: float,
    ) -> Float[torch.Tensor, "batch timesteps nodes features"]:
        """
        Perform a single DDIM sampling step from t to t_next.
        
        DDIM sampling equation:
        x_{t-1} = sqrt(α_{t-1}) * x0_pred + 
                  sqrt(1 - α_{t-1} - σ_t^2) * eps_pred +
                  σ_t * noise
        
        where σ_t = eta * sqrt((1 - α_{t-1}) / (1 - α_t)) * sqrt(1 - α_t / α_{t-1})
        
        Args:
            x_t: Current noisy sample
            t: Current timestep
            t_next: Next timestep (smaller than t)
            x0_pred: Predicted clean sample
            eps_pred: Predicted noise
            eta: Stochasticity parameter (0 = deterministic)
            
        Returns:
            x_{t_next}: Sample at next timestep
        """
        # Get alpha values at t and t_next.
        alpha_bar_t = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)

        # t_next = -1 is a virtual endpoint with alpha_bar_next = 1.0 (clean sample).
        alpha_bar_t_next = torch.ones_like(alpha_bar_t)
        valid_next = t_next >= 0
        if valid_next.any():
            alpha_bar_t_next[valid_next] = self._gather(self.alphas_cumprod, t_next[valid_next]).view(-1, 1, 1, 1)
        
        # Compute sigma (stochasticity)
        one_minus_alpha_bar_t = (1 - alpha_bar_t).clamp_min(1e-12)
        sigma_term1 = ((1 - alpha_bar_t_next).clamp_min(0.0) / one_minus_alpha_bar_t).clamp_min(0.0)
        sigma_term2 = (1 - alpha_bar_t / alpha_bar_t_next).clamp_min(0.0)
        sigma = (
            eta * 
            torch.sqrt(sigma_term1) *
            torch.sqrt(sigma_term2)
        )
        
        # Predicted direction pointing to x_t
        pred_dir_coeff = torch.sqrt((1 - alpha_bar_t_next - sigma ** 2).clamp_min(0.0))
        pred_dir = pred_dir_coeff * eps_pred
        
        # Compute x_{t_next}
        x_t_next = torch.sqrt(alpha_bar_t_next.clamp_min(0.0)) * x0_pred + pred_dir
        
        # Add stochastic noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_t_next = x_t_next + sigma * noise
        
        return x_t_next
    
    def get_speedup(self) -> float:
        """Return the sampling speedup factor compared to DDPM."""
        return self.num_timesteps / self.sampling_timesteps
