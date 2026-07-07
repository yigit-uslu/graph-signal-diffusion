# datasets/cifar10/dataset.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset
# from torch.utils.data import Dataset
from torch_geometric.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from graph_signal_diffusion.datasets import DATASET_REGISTRY
from graph_signal_diffusion.datasets.base import DatasetConfig
from graph_signal_diffusion.datasets.sp100.dataset import SP100Stocks, StocksDataDiffusion
from graph_signal_diffusion.datasets.sp100.utils_plot import plot_batched_data

from graph_signal_diffusion.datasets.normalizer import Normalizer


import logging
logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("sp100")
class SP100Builder:
    def __init__(self):
        self.normalizer: Optional[Normalizer] = None
        
    def build_datasets(self, cfg: DatasetConfig) -> Dict[str, Dataset]:
        # kwargs = cfg.kwargs or {}

        logger.info("Building SP100 datasets with configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
        kwargs = cfg.get("kwargs", {})
        past_window = int(cfg.get("past_window", 25))
        future_window = int(cfg.get("future_window", 1))
        target_column_name = cfg.get("target_column_name", "DailyLogReturn")
        corr_threshold = float(cfg.get("corr_threshold", 0.7)) if cfg.get("temporal_correlation_graph", False) else None
        pool_ratio = float(cfg.get("pool_ratio", 0.5))   

        logger.info("Loading SP100 stock data...")
        values = pd.read_csv(f'{cfg.root}/raw/values.csv').set_index(['Symbol', 'Date'])
        logger.info(f"Values head:\n{values.head()}")

        assert len(values.index.get_level_values('Symbol').unique()) == 100, "Expected 100 stocks, got {}".format(len(values.index.get_level_values('Symbol').unique()))

        # Assert there is the same number of dates for each stock
        assert all(values.index.get_level_values('Symbol').value_counts() == len(values.index.get_level_values('Date').unique())), "Not all stocks have the same number of dates."

        # Create full dataset
        dataset = SP100Stocks(root=cfg.root,
                            values_file_name="values.csv",
                            adj_file_name="adj.npy",
                            past_window=past_window,
                            future_window=future_window,
                            target_column_name=target_column_name,
                            corr_threshold=corr_threshold,
                            pool_ratio=pool_ratio
                            )
        
        logger.info(f"SP100Stocks dataset: {dataset}")
        logger.info(f"SP100Stocks dataset[0]: {dataset[0]}")
        logger.info(f"SP100Stocks dataset[-1]: {dataset[-1]}")


        ##### Dataset-split logic #####
        dataset_split_strategy = cfg.get("dataset_split_strategy", "chronological")
        train_dataset_fraction = float(cfg.get("train_dataset_fraction", 0.8))

        if dataset_split_strategy == "chronological":
            train_idx = torch.arange(0, int(len(dataset) * train_dataset_fraction))
            val_idx = torch.arange(train_idx[-1] + 1,
                                int(len(dataset) * (train_dataset_fraction + (1 - train_dataset_fraction) / 2))
            )
            test_idx = torch.arange(val_idx[-1] + 1,
                                    len(dataset)
                                    )
            # Last (1-f)/2 fraction of training data — immediately before val,
            # minimising distributional / volatility shift vs the val split.
            train_val_idx = torch.arange(
                int(len(dataset) * (train_dataset_fraction - (1 - train_dataset_fraction) / 2)),
                int(len(dataset) * train_dataset_fraction),
            )

            assert len(train_idx) > 0, "Training set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
            assert len(val_idx) > 0, "Validation set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
            assert len(test_idx) > 0, "Test set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
            assert len(train_val_idx) > 0, "Train-Val set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."


        elif dataset_split_strategy == "random":
            logger.info("Random dataset split strategy not yet implemented.")
            pass
            # train_idx, val_idx, test_idx, train_val_idx = \
            #     split_dataset_indices(dataset_indices = list(range(len(dataset))),
            #                         arg_groups = arg_groups
            #     )
        else:
            raise ValueError(f"Dataset split strategy {dataset_split_strategy} not recognized. Should be one of ['chronological', 'random'].")
        

        train_dataset = dataset.index_select(train_idx)

        val_dataset = dataset.index_select(val_idx)
        test_dataset = dataset.index_select(test_idx)
        train_val_dataset = dataset.index_select(train_val_idx)
        # val_dataset = dataset.index_select(val_idx[::future_window])
        # test_dataset = dataset.index_select(test_idx[::future_window])
        # train_val_dataset = dataset.index_select(train_val_idx[::future_window])

        logger.info(f"Train dataset: {len(train_dataset)} / {len(dataset)} samples.")
        logger.info(f"Validation dataset: {len(val_dataset)} / {len(dataset)} samples with indices from {val_idx[0]} to {val_idx[-1]}.")
        logger.info(f"Test dataset: {len(test_dataset)} / {len(dataset)} samples with indices from {test_idx[0]} to {test_idx[-1]}.")
        logger.info(f"Train-Val dataset: {len(train_val_dataset)} / {len(dataset)} samples with indices from {train_val_idx[0]} to {train_val_idx[-1]}.")


        # logger.info(f"Validation dataset: {len(val_dataset)} / {len(dataset)} samples with {future_window} delta timesteps in-between.")
        # logger.info(f"Test dataset: {len(test_dataset)} / {len(dataset)} samples with {future_window} delta timesteps in-between.")
        # logger.info(f"Train-Val dataset: {len(train_val_dataset)} / {len(dataset)} samples with {future_window} delta timesteps in-between.")



        ### Fit dataset normalizer on training data ###
        ################### BEGIN ###################
        normalize = cfg.get("normalize", False)
        if normalize: # fit normalizer if enabled
            normalizer_path = Path(cfg.root) / "processed" / "normalizer.json"
            
            if normalizer_path.exists():
                logger.info(f"📂 Loading normalizer from {normalizer_path}")
                self.normalizer = Normalizer.load(normalizer_path)
            else:
                logger.info("🔧 Fitting normalizer on training data...")
                normalize_method = cfg.get("normalize_method", "standardize")
                normalize_dim = cfg.get("normalize_dim", None)
                # batch_size = cfg.get("batch_size", 32)
                
                self.normalizer = self._fit_normalizer(
                    train_dataset,
                    method=normalize_method,
                    dim=tuple(normalize_dim) if normalize_dim else None,
                    batch_size=32   # Optional: only if using batched data collection
                )
                self.normalizer.save(normalizer_path)
                logger.info(f"💾 Saved normalizer to {normalizer_path}")
            
            # Inject into all datasets
            for split_name, split_dataset in [
                ("train", train_dataset),
                ("val", val_dataset),
                ("test", test_dataset),
                ("train-val", train_val_dataset)
            ]:
                if hasattr(split_dataset, 'dataset'):  # index_select returns Subset
                    if hasattr(split_dataset.dataset, 'set_normalizer'):
                        split_dataset.dataset.set_normalizer(self.normalizer)
                else:  # Direct dataset
                    if hasattr(split_dataset, 'set_normalizer'):
                        split_dataset.set_normalizer(self.normalizer)
                logger.info(f"   ✅ Normalizer injected into {split_name}")

        ################### END ###################

        # Store reference to train dataset for getting dataset_info later
        self._train_dataset = train_dataset

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "train-val": train_val_dataset
        }

    def get_dataset_info(self) -> Optional[Dict]:
        """
        Get dataset info dict containing metadata like log_return_scale_factors.
        
        Returns:
            Dataset info dictionary or None if not available.
            Handles both direct datasets and Subset wrappers.
        """
        if not hasattr(self, '_train_dataset') or self._train_dataset is None:
            return None
        
        dataset = self._train_dataset
        # Handle Subset wrapper (from index_select)
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        
        # Get info from first sample
        if len(dataset) > 0 and hasattr(dataset[0], 'info'):
            return dataset[0].info
        
        return None


    def _fit_normalizer(
            self,
            train_dataset: Dataset,
            method: str,
            dim: Optional[Tuple[int, ...]],
            batch_size: int = 32 # just for memory efficiency
        ) -> Normalizer:

        """Fit normalizer on training data."""

        # Access underlying dataset if it's a Subset
        actual_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
        
        # Temporarily disable normalization
        if hasattr(actual_dataset, '_normalize_targets'):
            actual_dataset._normalize_targets = False


        # Collect data directly by iterating over the dataset
        all_y = []
        logger.info("📊 Collecting training samples for normalization...")
        logger.info(f"   Total samples: {len(train_dataset)}")

        for i in range(len(train_dataset)):
            data = train_dataset[i]
            all_y.append(data.y.cpu())

            if (i + 1) % 100 == 0:
                logger.info(f"   Collected {i + 1}/{len(train_dataset)} samples...")
        
        # if isinstance(train_dataset, Subset):
        #     indices = train_dataset.indices
        # else:
        #     indices = range(len(train_dataset))
        
        # # # Option 1: Collect all at once (simple, works for small datasets)
        # # for idx in indices:
        # #     all_y.append(train_dataset[idx].y.cpu())
        
        # # Option 2: Collect in batches (memory efficient for large datasets)
        # for i in range(0, len(indices), batch_size):
        #     batch_indices = indices[i:i+batch_size]
        #     batch_y = [train_dataset[idx].y.cpu() for idx in batch_indices]
        #     all_y.extend(batch_y)
            
        #     if i % (batch_size * 10) == 0:
        #         print(f"   Collected {len(all_y)} samples...")


        all_y = torch.cat(all_y, dim=0)
        logger.info(f"   Collected {all_y.shape[0]} samples.")
        logger.info(f"   Shape: {all_y.shape}")
        logger.info(f"   Original - Mean: {all_y.mean():.4f}, Std: {all_y.std():.4f}")
        
        # Fit normalizer
        normalizer = Normalizer(method=method, dim=dim)
        normalizer.fit(all_y)
        
        logger.info(f"✅ Normalizer fitted: {normalizer}")
        if method == "standardize":
            logger.info(f"   Mean shape: {normalizer.mean.shape}")
            logger.info(f"   Std shape: {normalizer.std.shape}")
        
        # Re-enable normalization
        if hasattr(actual_dataset, '_normalize_targets'):
            actual_dataset._normalize_targets = True

        return normalizer
        
    

    def build_loaders(self, cfg: DatasetConfig, datasets: Dict[str, Dataset], accelerator=None) -> Dict[str, DataLoader]:
        # dataset-specific collate can live here if needed
        g = torch.Generator()
        g.manual_seed(0)

        x_batch_size = int(cfg.get("x_batch_size", 1))

        if x_batch_size > 1:
            # Effective batch size for training
            batch_size_train = x_batch_size * cfg.batch_size
            # epoch_size_train = batch_size_train * 10  # e.g., 10 batches per epoch
            epoch_size_train = len(datasets["train"]) * x_batch_size  # One epoch covers the whole dataset

            # Create weights (uniform for simplicity)
            weights = torch.ones(len(datasets["train"]))
            # Create sampler - key point: num_samples can be any value
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=epoch_size_train,  # e.g., 1500 samples per epoch
                replacement=True,  # This is crucial!
                generator=g
            )

            # x_batch_size = int(cfg.get("x_batch_size", 1))
            # batch_size_train = x_batch_size * cfg.batch_size

            logger.info("Using weighted random sampler for training dataloader.")
            logger.info(f"Train batch size set to {batch_size_train}.")
            train_loader = DataLoader(
                datasets["train"],
                batch_size=batch_size_train,
                sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                drop_last=cfg.drop_last,
                follow_batch=["y"],
                exclude_keys=["close_price", "close_price_y", "timestamp", "info"]
            )

        else:
            logger.info("Using standard sequential sampler for training dataloader with shuffling.")
            logger.info(f"Train batch size set to {cfg.batch_size}.")
            train_loader = DataLoader(
                datasets["train"],
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                drop_last=cfg.drop_last,
                generator=g,
                follow_batch=["y"],
                exclude_keys=["close_price", "close_price_y", "timestamp", "info"]
            )


        # batch_size_train_val = len(datasets["train-val"])
        # batch_size_val = min(2, len(datasets["val"]) // 6)
        # batch_size_test = min(2, len(datasets["test"]) // 6)
        batch_size_val = cfg.get("batch_size_val", min(2, len(datasets["val"]) // 6))
        batch_size_test, batch_size_train_val = batch_size_val, batch_size_val

        logger.info(f"Validation batch size set to {batch_size_val}.")
        logger.info(f"Train-Val batch size set to {batch_size_train_val}.")
        logger.info(f"Test batch size set to {batch_size_test}.")

        train_val_loader = DataLoader(
            datasets["train-val"],
            batch_size=batch_size_train_val, 
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory, 
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=cfg.drop_last,
            follow_batch=["y"],
            exclude_keys=["info"]
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=cfg.drop_last,
            follow_batch=["y"],
            exclude_keys=["info"]
        )

        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=cfg.drop_last,
            follow_batch=["y"],
            exclude_keys=["info"]
        )

        # Prepare with Accelerator if provided
        if accelerator:
            train_loader, val_loader, test_loader, train_val_loader = accelerator.prepare(
                train_loader, val_loader, test_loader, train_val_loader)
            
        # Generate plots for data visualization if not already present
        plots_dir = f"{cfg.root}/plots" 
        if os.path.exists(plots_dir) and os.listdir(plots_dir):
            logger.info(f"Plots directory {plots_dir} already exists and is not empty. Skipping plot generation.")
        else:
            logger.info(f"Generating plots in {plots_dir} ...")
            # Visualize dataloader evolution for test dataset
            data = next(iter(test_loader))
            num_stocks = data.x.shape[0] // data.num_graphs
            num_features = data.x.shape[-1]
            stocks_idx = np.random.choice(num_stocks, 4)

            if isinstance(test_loader.dataset, Subset):
                # print("Subset detected for test_loader.dataset")
                # print("Test_loader.dataset: ", test_loader.dataset)
                # print("Underlying dataset: ", test_loader.dataset.dataset)
                target_column_name = test_loader.dataset.dataset[0].info["Target"] # target_column_name
                info_features = test_loader.dataset.dataset[0].info['Features']
                logger.info(f"Extracted target_column_name = {target_column_name} and features = {info_features} from underlying dataset.")

            else:
                # print("Direct dataset detected for test_loader.dataset")
                # print("Underlying dataset: ", test_loader.dataset)
                target_column_name = test_loader.dataset[0].info["Target"]
                info_features = test_loader.dataset[0].info['Features']
                logger.info(f"Extracted target_column_name = {target_column_name} and features = {info_features} directly.")
            
            # target_column_name = test_loader.dataset.dataset.target_column_name if isinstance(test_loader.dataset, Subset) else test_loader.dataset.target_column_name
            # info_features = test_loader.dataset.dataset.info['Features'] if isinstance(test_loader.dataset, Subset) else test_loader.dataset.info['Features']
            
            for plot_target in ["Closing_Price", ("y", target_column_name), "Closing_Price_y"] + \
                [(f'x[{i}]', info_features[i]) for i in range(num_features)]:
                plot_batched_data(
                    data = data.clone(),
                    stocks_idx = stocks_idx,
                    plot_target=plot_target,
                    save_dir = f"{cfg.root}/plots",
                )

        return {"train": train_loader, "val": val_loader, "test": test_loader, "train-val": train_val_loader}






# @DATASET_REGISTRY.register("sp100")
# class SP100Builder:
#     def build_datasets(self, cfg: DatasetConfig) -> Dict[str, Dataset]:
#         # kwargs = cfg.kwargs or {}
#         kwargs = cfg.get("kwargs", {})
#         past_window = int(cfg.get("past_window", 25))
#         future_window = int(cfg.get("future_window", 1))
#         target_column_name = cfg.get("target_column_name", "NormClose")
#         corr_threshold = float(cfg.get("corr_threshold", 0.7)) if cfg.get("temporal_correlation_graph", False) else None
#         pool_ratio = float(cfg.get("pool_ratio", 0.5))   

#         print("Loading SP100 stock data...")
#         values = pd.read_csv(f'{cfg.root}/raw/values.csv').set_index(['Symbol', 'Date'])
#         values.head()

#         assert len(values.index.get_level_values('Symbol').unique()) == 100, "Expected 100 stocks, got {}".format(len(values.index.get_level_values('Symbol').unique()))

#         # Assert there is the same number of dates for each stock
#         assert all(values.index.get_level_values('Symbol').value_counts() == len(values.index.get_level_values('Date').unique())), "Not all stocks have the same number of dates."


#         dataset = SP100Stocks(root=cfg.root,
#                             values_file_name="values.csv",
#                             adj_file_name="adj.npy",
#                             past_window=past_window,
#                             future_window=future_window,
#                             target_column_name=target_column_name,
#                             corr_threshold=corr_threshold,
#                             pool_ratio=pool_ratio
#                             )
        
#         print("SP100Stocks dataset: ", dataset)
#         print("SP100Stocks dataset[0]: ", dataset[0])
#         print("SP100Stocks dataset[-1]: ", dataset[-1])



#         ##### Dataset-split logic #####
#         dataset_split_strategy = cfg.get("dataset_split_strategy", "chronological")
#         train_dataset_fraction = float(cfg.get("train_dataset_fraction", 0.8))

#         if dataset_split_strategy == "chronological":
#             train_idx = torch.arange(0, int(len(dataset) * train_dataset_fraction))
#             val_idx = torch.arange(train_idx[-1] + 1,
#                                 int(len(dataset) * (train_dataset_fraction + (1 - train_dataset_fraction) / 2))
#             )
#             test_idx = torch.arange(val_idx[-1] + 1,
#                                     len(dataset)
#                                     )
#             train_val_idx = torch.arange(0, int(len(dataset) * (1 - train_dataset_fraction) / 2))

#             assert len(train_idx) > 0, "Training set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
#             assert len(val_idx) > 0, "Validation set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
#             assert len(test_idx) > 0, "Test set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
#             assert len(train_val_idx) > 0, "Train-Val set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."


#         elif dataset_split_strategy == "random":
#             print("Random dataset split strategy not yet implemented.")
#             pass
#             # train_idx, val_idx, test_idx, train_val_idx = \
#             #     split_dataset_indices(dataset_indices = list(range(len(dataset))),
#             #                         arg_groups = arg_groups
#             #     )
#         else:
#             raise ValueError(f"Dataset split strategy {dataset_split_strategy} not recognized. Should be one of ['chronological', 'random'].")
        

#         train_dataset = dataset.index_select(train_idx)
#         val_dataset = dataset.index_select(val_idx[::future_window])
#         test_dataset = dataset.index_select(test_idx[::future_window])
#         train_val_dataset = dataset.index_select(train_val_idx[::future_window])

#         print(f"Train dataset: {len(train_dataset)} / {len(dataset)} samples.")
#         print(f"Validation dataset: {len(val_dataset)} / {len(dataset)} samples with {future_window} delta timesteps in-between.")
#         print(f"Test dataset: {len(test_dataset)} / {len(dataset)} samples with {future_window} delta timesteps in-between.")
#         print(f"Train-Val dataset: {len(train_val_dataset)} / {len(dataset)} samples with {future_window} delta timesteps in-between.")

#         return {"train": train_dataset, "val": val_dataset, "test": test_dataset, "train-val": train_val_dataset}
    

#     def build_loaders(self, cfg: DatasetConfig, datasets: Dict[str, Dataset], accelerator=None) -> Dict[str, DataLoader]:
#         # dataset-specific collate can live here if needed
#         g = torch.Generator()
#         g.manual_seed(0)

#         if cfg.batch_size > 1:
#             # Create weights (uniform for simplicity)
#             weights = torch.ones(len(datasets["train"]))
#             # Create sampler - key point: num_samples can be any value
#             sampler = WeightedRandomSampler(
#                 weights=weights,
#                 num_samples=cfg.batch_size * 10,  # e.g., 1500 samples per epoch
#                 replacement=True,  # This is crucial!
#                 generator=g
#             )

#             # kwargs = cfg.get("kwargs", {})
#             x_batch_size = int(cfg.get("x_batch_size", 1))
#             batch_size_train = x_batch_size * cfg.batch_size

#             print("Using weighted random sampler for training dataloader.")
#             train_loader = DataLoader(
#                 datasets["train"],
#                 batch_size=batch_size_train,
#                 sampler=sampler,
#                 num_workers=cfg.num_workers,
#                 pin_memory=cfg.pin_memory,
#                 persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
#                 drop_last=cfg.drop_last,
#                 follow_batch=["y"],
#                 exclude_keys=["close_price", "close_price_y", "timestamp"]
#             )

#         else:
#             print("Using standard sequential sampler for training dataloader with shuffling.")
#             train_loader = DataLoader(
#                 datasets["train"],
#                 batch_size=cfg.batch_size,
#                 shuffle=True,
#                 num_workers=cfg.num_workers,
#                 pin_memory=cfg.pin_memory,
#                 persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
#                 drop_last=cfg.drop_last,
#                 generator=g,
#                 follow_batch=["y"],
#                 exclude_keys=["close_price", "close_price_y", "timestamp"]
#             )



#         batch_size_train_val = len(datasets["train-val"])
#         batch_size_val = len(datasets["val"])
#         batch_size_test = len(datasets["test"])

#         train_val_loader = DataLoader(
#             datasets["train-val"],
#             batch_size=batch_size_train_val, 
#             shuffle=False,
#             num_workers=cfg.num_workers,
#             pin_memory=cfg.pin_memory, 
#             persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
#             drop_last=cfg.drop_last,
#             follow_batch=["y"]
#         )

#         val_loader = DataLoader(
#             datasets["val"],
#             batch_size=batch_size_val,
#             shuffle=False, 
#             num_workers=cfg.num_workers,
#             pin_memory=cfg.pin_memory,
#             persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
#             drop_last=cfg.drop_last,
#             follow_batch=["y"]
#         )

#         test_loader = DataLoader(
#             datasets["test"],
#             batch_size=batch_size_test,
#             shuffle=False,
#             num_workers=cfg.num_workers,
#             pin_memory=cfg.pin_memory,
#             persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
#             drop_last=cfg.drop_last,
#             follow_batch=["y"]
#         )

#         # Prepare with Accelerator if provided
#         if accelerator:
#             train_loader, val_loader, test_loader, train_val_loader = accelerator.prepare(
#                 train_loader, val_loader, test_loader, train_val_loader)
            
#         # Generate plots for data visualization if not already present
#         plots_dir = f"{cfg.root}/plots" 
#         if os.path.exists(plots_dir) and os.listdir(plots_dir):
#             print(f"Plots directory {plots_dir} already exists and is not empty. Skipping plot generation.")
#         else:
#             print(f"Generating plots in {plots_dir} ...")
#             # Visualize dataloader evolution for test dataset
#             data = next(iter(test_loader))
#             stocks_idx = np.random.choice(100, 4)
#             num_features = data.x.shape[1]

#             target_column_name = test_loader.dataset.dataset.target_column_name if isinstance(test_loader.dataset, Subset) else test_loader.dataset.target_column_name
#             info_features = test_loader.dataset.dataset.info['Features'] if isinstance(test_loader.dataset, Subset) else test_loader.dataset.info['Features']
            
#             for plot_target in ["Closing_Price", ("y", target_column_name), "Closing_Price_y"] + \
#                 [(f'x[{i}]', info_features[i]) for i in range(num_features)]:
#                 plot_batched_data(
#                     data = data.clone(),
#                     stocks_idx = stocks_idx,
#                     plot_target=plot_target,
#                     save_dir = f"{cfg.root}/plots",
#                 )

#         return {"train": train_loader, "val": val_loader, "test": test_loader, "train-val": train_val_loader}

