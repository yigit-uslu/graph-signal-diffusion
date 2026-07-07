# ./datasets/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

@dataclass
class DatasetConfig:
    name: str
    root: str = "./data"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True
    # dataset-specific kwargs live here:
    kwargs: Dict[str, Any] = None

class DatasetBuilder(Protocol):
    """
    A dataset "plugin" must implement this.
    Keep the interface narrow and stable.
    """
    def build_datasets(self, cfg: DatasetConfig) -> Dict[str, Dataset]: ...
    def build_loaders(self, cfg: DatasetConfig, datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]: ...
