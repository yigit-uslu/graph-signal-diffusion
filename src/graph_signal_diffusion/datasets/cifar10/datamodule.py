# datasets/cifar10/dataset.py
from __future__ import annotations
from typing import Dict
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms as T

from graph_signal_diffusion.datasets import DATASET_REGISTRY
from graph_signal_diffusion.datasets.base import DatasetConfig

@DATASET_REGISTRY.register("cifar10")
class CIFAR10Builder:
    def build_datasets(self, cfg: DatasetConfig) -> Dict[str, Dataset]:
        # kwargs = cfg.kwargs or {}
        kwargs = cfg.get("kwargs", {})
        image_size = int(kwargs.get("image_size", 32))

        tf_train = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        tf_eval = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

        train = tv_datasets.CIFAR10(root=cfg.root, train=True, download=True, transform=tf_train)
        val = tv_datasets.CIFAR10(root=cfg.root, train=False, download=True, transform=tf_eval)

        return {"train": train, "val": val}
    

    def build_loaders(self, cfg: DatasetConfig, datasets: Dict[str, Dataset], accelerator=None) -> Dict[str, DataLoader]:
        # dataset-specific collate can live here if needed
        g = torch.Generator()
        g.manual_seed(0)

        train_loader = DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=cfg.drop_last,
            generator=g,
        )
        val_loader = DataLoader(
            datasets["val"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=False,
        )

        # Prepare with Accelerator if provided
        if accelerator:
            train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

        return {"train": train_loader, "val": val_loader}
