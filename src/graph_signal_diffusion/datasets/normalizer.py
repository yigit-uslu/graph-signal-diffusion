from __future__ import annotations
from typing import Optional, Literal
import torch
from pathlib import Path
import json
from abc import ABC


class Normalizer(ABC):
    """Handles normalization/denormalization for diffusion models."""
    
    def __init__(
        self,
        method: Literal["standardize", "minmax", "none"] = "standardize",
        dim: Optional[tuple] = None,
        eps: float = 1e-8,
    ):
        self.method = method
        self.dim = dim
        self.eps = eps
        
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.min: Optional[torch.Tensor] = None
        self.max: Optional[torch.Tensor] = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> Normalizer:
        """Compute normalization statistics."""
        if self.method == "none":
            self.fitted = True
            return self
        
        with torch.no_grad():
            if self.method == "standardize":
                if self.dim is not None:
                    self.mean = data.mean(dim=self.dim, keepdim=True)
                    self.std = data.std(dim=self.dim, keepdim=True) + self.eps
                else:
                    self.mean = data.mean()
                    self.std = data.std() + self.eps
                    
            elif self.method == "minmax":
                if self.dim is not None:
                    self.min = data.amin(dim=self.dim, keepdim=True)
                    self.max = data.amax(dim=self.dim, keepdim=True)
                else:
                    self.min = data.min()
                    self.max = data.max()
        
        self.fitted = True
        return self
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data."""
        if not self.fitted or self.method == "none":
            return data
        
        if self.method == "standardize":
            return (data - self.mean.to(data.device)) / self.std.to(data.device)
        elif self.method == "minmax":
            return (data - self.min.to(data.device)) / (
                self.max.to(data.device) - self.min.to(data.device) + self.eps
            )
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data."""
        if not self.fitted or self.method == "none":
            return data
        
        if self.method == "standardize":
            return data * self.std.to(data.device) + self.mean.to(data.device)
        elif self.method == "minmax":
            return data * (
                self.max.to(data.device) - self.min.to(data.device) + self.eps
            ) + self.min.to(data.device)
    
    def save(self, path: str | Path) -> None:
        """Save to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "method": self.method,
            "dim": list(self.dim) if self.dim else None,
            "eps": self.eps,
            "fitted": self.fitted,
        }
        
        if self.mean is not None:
            state["mean"] = self.mean.cpu().tolist()
        if self.std is not None:
            state["std"] = self.std.cpu().tolist()
        if self.min is not None:
            state["min"] = self.min.cpu().tolist()
        if self.max is not None:
            state["max"] = self.max.cpu().tolist()
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        """Load from JSON."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        normalizer = cls(
            method=state["method"],
            dim=tuple(state["dim"]) if state["dim"] else None,
            eps=state["eps"]
        )
        normalizer.fitted = state["fitted"]
        
        if "mean" in state:
            normalizer.mean = torch.tensor(state["mean"])
        if "std" in state:
            normalizer.std = torch.tensor(state["std"])
        if "min" in state:
            normalizer.min = torch.tensor(state["min"])
        if "max" in state:
            normalizer.max = torch.tensor(state["max"])
        
        return normalizer
    
    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return f"Normalizer(method={self.method}, {status})"