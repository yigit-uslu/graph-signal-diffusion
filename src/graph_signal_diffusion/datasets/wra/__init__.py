"""Wireless Resource Allocation dataset module.

Provides dataset classes and configuration management for power allocation
diffusion training.
"""

from .dataset import WRADataset, WirelessDataDiffusion
from .datamodule import WRABuilder  # noqa: F401
from .channel import WirelessChannel, WirelessChannelV2, WirelessChannelV3
from .configs import dataset_name_to_alias

__all__ = [
    'WRADataset',
    'WirelessDataDiffusion',
    'WirelessChannel',
    'WirelessChannelV2',
    'WirelessChannelV3',
    'WRABuilder',
    'dataset_name_to_alias',
]
