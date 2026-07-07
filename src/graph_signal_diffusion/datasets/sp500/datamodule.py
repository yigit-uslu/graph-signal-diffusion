# datasets/sp500/dataset.py
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
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
from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks
from graph_signal_diffusion.datasets.sp500.utils_plot import plot_batched_data
from graph_signal_diffusion.datasets.normalizer import Normalizer
from graph_signal_diffusion.datasets.replicated_dataset import ReplicatedDataset
from graph_signal_diffusion.datasets.replicated_sampler import ReplicatedGroupedBatchSampler
from graph_signal_diffusion.datasets.replicated_sampler import ReplicatedGroupedBatchSampler

import logging 
log = logging.getLogger(__name__)


@DATASET_REGISTRY.register("sp500")
@DATASET_REGISTRY.register("sp500_cleaned")  # Alias for cleaned data variant
class SP500Builder:
    def __init__(self):
        self.normalizer: Optional[Normalizer] = None
        
        # NEW: Per-stock standardization metadata (Solution 1)
        self.per_stock_stats: Optional[Dict] = None
        self.sp500_scale_factor: float = 100.0  # SP500 DailyLogReturn is scaled by 100x
        self.feature_names: list = []
        self.standardized_features: list = []
        self.target_column_name: str = "DailyLogReturn"  # Default, overridden in build_datasets()
        
    def build_datasets(self, cfg: DatasetConfig) -> Dict[str, Dataset]:
        # kwargs = cfg.kwargs or {}
        kwargs = cfg.get("kwargs", {})
        past_window = int(cfg.get("past_window", 25))
        future_window = int(cfg.get("future_window", 1))
        target_column_name = cfg.get("target_column_name", "DailyLogReturn")
        self.target_column_name = target_column_name  # Store for get_dataset_info()
        pool_ratio = float(cfg.get("pool_ratio", 0.5))
        top_k = cfg.get("top_k", None)
        min_degree = cfg.get("min_degree", None)   

        log.info("Loading SP500 stock data...")
        values = pd.read_csv(f'{cfg.root}/raw/values.csv').set_index(['Symbol', 'Date'])
        values.head()

        # Note: SP500 may have slightly fewer than 500 stocks due to data availability
        num_stocks = len(values.index.get_level_values('Symbol').unique())
        log.info(f"Loaded data for {num_stocks} stocks")
        assert num_stocks >= 400, f"Expected at least 400 stocks, got {num_stocks}"

        # Assert there is the same number of dates for each stock
        assert all(values.index.get_level_values('Symbol').value_counts() == len(values.index.get_level_values('Date').unique())), "Not all stocks have the same number of dates."

        # NEW: Dynamically extract feature names from CSV (after dropping Close)
        # This handles different column orders between SP100 and SP500
        self.feature_names = values.drop(columns=['Close']).columns.tolist()
        log.info(f"📋 Feature columns (after dropping Close): {self.feature_names}")
        
        # Define which features to standardize
        # Exclude NormClose and RSI (both already normalized)
        # RSI is in [0,1] range, NormClose is pre-standardized
        exclude_from_standardization = {'NormClose', 'RSI'}
        self.standardized_features = [f for f in self.feature_names if f not in exclude_from_standardization]
        log.info(f"🔧 Features to be standardized: {self.standardized_features}")
        
        # Identify features needing log-transform before standardization
        self.log_transform_features = ['Volume']  # Volume is heavily skewed
        log.info(f"📊 Features to log-transform: {self.log_transform_features}")
        
        # Compute per-stock standardization statistics
        self.per_stock_stats = self._compute_per_stock_stats(values)
        
        # Create full dataset
        # Extract dynamic graph parameters
        dynamic_graph_period = cfg.get("dynamic_graph_period", None)
        dynamic_graph_top_k = cfg.get("dynamic_graph_top_k", None)
        dynamic_graph_min_degree = cfg.get("dynamic_graph_min_degree", None)
        dynamic_graph_method = str(cfg.get("dynamic_graph_method", "pearson"))
        dynamic_graph_edge_weight_threshold = cfg.get("dynamic_graph_edge_weight_threshold", None)
        dynamic_graph_edge_budget_ratio = cfg.get("dynamic_graph_edge_budget_ratio", None)

        standardize_target_in_x_for_revin = bool(
            cfg.get('standardize_target_in_x_for_revin', False)
        )

        dataset = SP500Stocks(root=cfg.root,
                            values_file_name="values.csv",
                            adj_file_name="adj.npy",
                            past_window=past_window,
                            future_window=future_window,
                            target_column_name=target_column_name,
                            pool_ratio=pool_ratio,
                            per_stock_stats=self.per_stock_stats,
                            feature_names=self.feature_names,
                            standardized_features=self.standardized_features,
                            log_transform_features=self.log_transform_features,
                            sp500_scale_factor=self.sp500_scale_factor,
                            top_k=top_k,
                            min_degree=min_degree,
                            edge_weight_norm=str(cfg.get("edge_weight_norm", "none")),
                            spectral_normalize_edge_weights=bool(
                                cfg.get("spectral_normalize_edge_weights", False)
                            ),
                            dynamic_graph_period=int(dynamic_graph_period) if dynamic_graph_period is not None else None,
                            dynamic_graph_top_k=int(dynamic_graph_top_k) if dynamic_graph_top_k is not None else None,
                            dynamic_graph_min_degree=int(dynamic_graph_min_degree) if dynamic_graph_min_degree is not None else None,
                            dynamic_graph_method=dynamic_graph_method,
                            dynamic_graph_edge_weight_threshold=(
                                float(dynamic_graph_edge_weight_threshold)
                                if dynamic_graph_edge_weight_threshold is not None else None
                            ),
                            dynamic_graph_edge_budget_ratio=(
                                float(dynamic_graph_edge_budget_ratio)
                                if dynamic_graph_edge_budget_ratio is not None else None
                            ),
                            standardize_target_in_x_for_revin=standardize_target_in_x_for_revin,
                            )
        
        # log.info(f"SP500Stocks dataset: {dataset}")
        # log.info("SP500Stocks dataset[0]: %s", dataset[0])
        # log.info("SP500Stocks dataset[-1]: %s", dataset[-1])


        ##### Dataset-split logic #####
        dataset_split_strategy = cfg.get("dataset_split_strategy", "chronological")
        train_dataset_fraction = float(cfg.get("train_dataset_fraction", 0.8))

        if dataset_split_strategy == "chronological":
            n_split_chunks = int(cfg.get("n_split_chunks", 1))
            assert n_split_chunks >= 1, f"n_split_chunks must be >= 1, got {n_split_chunks}"
            N = len(dataset)
            val_frac = (1 - train_dataset_fraction) / 2  # e.g. 0.1

            if n_split_chunks == 1:
                # Original single-chunk logic (fast path, identical to legacy behaviour)
                train_idx = torch.arange(0, int(N * train_dataset_fraction))
                val_idx = torch.arange(train_idx[-1] + 1,
                                       int(N * (train_dataset_fraction + val_frac)))
                test_idx = torch.arange(val_idx[-1] + 1, N)
                # Last val_frac of training data — immediately before val
                train_val_idx = torch.arange(
                    int(N * (train_dataset_fraction - val_frac)),
                    int(N * train_dataset_fraction),
                )
            else:
                chunk_size = N // n_split_chunks
                train_ids, val_ids, test_ids = [], [], []
                for c in range(n_split_chunks):
                    start = c * chunk_size
                    end = (start + chunk_size) if c < n_split_chunks - 1 else N
                    n = end - start
                    t_end = start + int(n * train_dataset_fraction)
                    v_end = start + int(n * (train_dataset_fraction + val_frac))
                    train_ids.append(torch.arange(start, t_end))
                    val_ids.append(torch.arange(t_end, v_end))
                    test_ids.append(torch.arange(v_end, end))
                train_idx = torch.cat(train_ids)
                val_idx = torch.cat(val_ids)
                test_idx = torch.cat(test_ids)
                # train-val = train portion of the last chunk (closest in time to its val)
                train_val_idx = train_ids[-1]
                log.info(
                    f"Interleaved chronological split: {n_split_chunks} chunks, "
                    f"chunk_size≈{chunk_size}, "
                    f"train={len(train_idx)}, val={len(val_idx)}, "
                    f"test={len(test_idx)}, train-val={len(train_val_idx)}"
                )

            # Optional: truncate training set to only the most recent N samples.
            # Val/test/train-val splits are unchanged.
            train_window_days = cfg.get("train_window_days", None)
            if train_window_days is not None:
                train_window_days = int(train_window_days)
                if train_window_days < len(train_idx):
                    log.info(f"Truncating training set: {len(train_idx)} -> {train_window_days} samples "
                             f"(keeping most recent {train_window_days} days)")
                    train_idx = train_idx[-train_window_days:]

            assert len(train_idx) > 0, "Training set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
            assert len(val_idx) > 0, "Validation set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
            assert len(test_idx) > 0, "Test set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."
            assert len(train_val_idx) > 0, "Train-Val set is empty after splitting. Please adjust the train_dataset_fraction or chunk_size."


        elif dataset_split_strategy == "random":
            log.info("Random dataset split strategy not yet implemented.")
            pass
            # train_idx, val_idx, test_idx, train_val_idx = \
            #     split_dataset_indices(dataset_indices = list(range(len(dataset))),
            #                         arg_groups = arg_groups
            #     )
        else:
            raise ValueError(f"Dataset split strategy {dataset_split_strategy} not recognized. Should be one of ['chronological', 'random'].")

        # ── Optional uniform-stride subsampling of the test set ─────────
        # When ``test_subsample_n`` is set, replace ``test_idx`` with a
        # deterministic ``np.linspace`` subsample of length K covering the
        # full test span (first and last test windows always included,
        # chunks 0..n_split_chunks-1 each contribute proportionally because
        # ``test_idx`` is already in chronological order).
        #
        # Use case: cheap-but-representative test coverage for expensive
        # baselines (e.g. diffusion at n_samples=50) where the full test
        # set would take many hours.  Without this, ``eval_splits.test=1``
        # would consume only the front of ``test_idx`` and bias the eval
        # toward chunk 0 (see datamodule docstring / Phase-2 sampling fix).
        test_subsample_n = cfg.get("test_subsample_n", None)
        if test_subsample_n is not None:
            test_subsample_n = int(test_subsample_n)
            if test_subsample_n <= 0:
                raise ValueError(
                    f"test_subsample_n must be a positive integer or null, "
                    f"got {test_subsample_n}"
                )
            if test_subsample_n < len(test_idx):
                # Linspace inclusive of both endpoints, then dedupe in case
                # rounding collapses adjacent picks at small K / boundaries.
                pick = np.linspace(
                    0, len(test_idx) - 1, num=test_subsample_n,
                ).round().astype(np.int64)
                pick = np.unique(pick)
                test_idx = test_idx[torch.from_numpy(pick)]
                log.info(
                    f"Subsampling test set: {len(test_idx)} unique windows "
                    f"(linspace, target={test_subsample_n})"
                )

                # Phased-estimator sanity check.
                #
                # The structural-metric phased pooling (Option A in
                # ``evaluation/structural_metrics.py``) groups windows by
                # ``start_index % future_window`` to undo stride-1
                # sliding-window overlap dilution.  If the linspace pick
                # produces a stride that is an exact multiple of
                # ``future_window``, all picks land in a single phase
                # modulo ``future_window`` and the phased average
                # degenerates to that one phase only.  Warn loudly so the
                # caller can adjust ``test_subsample_n``.
                future_window = int(cfg.get("future_window", 1) or 1)
                if future_window > 1:
                    phases = (test_idx.numpy() % future_window).astype(np.int64)
                    counts = np.bincount(phases, minlength=future_window)
                    n_empty = int((counts == 0).sum())
                    n_full = int((counts > 0).sum())
                    if n_empty > 0:
                        log.warning(
                            "test_subsample_n=%d produces a degenerate phase "
                            "distribution for the structural-metric phased "
                            "estimator: only %d/%d phases have any windows "
                            "(per-phase counts: %s, future_window=%d). "
                            "Picks land at stride that is (near-)multiple of "
                            "future_window. Adjust test_subsample_n (e.g. "
                            "by ±1) to recover full phase coverage.",
                            test_subsample_n, n_full, future_window,
                            counts.tolist(), future_window,
                        )
                    elif counts.max() > 2 * counts.min():
                        log.warning(
                            "test_subsample_n=%d produces an uneven phase "
                            "distribution (per-phase counts: %s, "
                            "future_window=%d). Some phases will dominate "
                            "the phased structural-metric average.",
                            test_subsample_n, counts.tolist(), future_window,
                        )
                    else:
                        log.info(
                            "Phased estimator coverage OK: per-phase counts "
                            "= %s (future_window=%d)",
                            counts.tolist(), future_window,
                        )
            else:
                log.info(
                    f"test_subsample_n={test_subsample_n} >= |test|="
                    f"{len(test_idx)}; keeping full test set."
                )

        train_dataset = dataset.index_select(train_idx)

        val_dataset = dataset.index_select(val_idx)
        test_dataset = dataset.index_select(test_idx)
        train_val_dataset = dataset.index_select(train_val_idx)
        # val_dataset = dataset.index_select(val_idx[::future_window])
        # test_dataset = dataset.index_select(test_idx[::future_window])
        # train_val_dataset = dataset.index_select(train_val_idx[::future_window])

        log.info(f"Train dataset: {len(train_dataset)} / {len(dataset)} samples.")
        log.info(f"Validation dataset: {len(val_dataset)} / {len(dataset)} samples with indices from {val_idx[0]} to {val_idx[-1]}.")
        log.info(f"Test dataset: {len(test_dataset)} / {len(dataset)} samples with indices from {test_idx[0]} to {test_idx[-1]}.")
        log.info(f"Train-Val dataset: {len(train_val_dataset)} / {len(dataset)} samples with indices from {train_val_idx[0]} to {train_val_idx[-1]}.")



        # Wrap evaluation datasets with ReplicatedDataset for multiple samples per input
        # Read from dataset config (which interpolates from trainer.n_samples_per_input via Hydra)
        n_eval_samples = int(cfg.get("n_samples_per_input", 1))
        if n_eval_samples > 1:
            log.info(f"\n🔁 Wrapping evaluation datasets with ReplicatedDataset (n_replicas={n_eval_samples})")
            val_dataset = ReplicatedDataset(val_dataset, n_replicas=n_eval_samples)
            test_dataset = ReplicatedDataset(test_dataset, n_replicas=n_eval_samples)
            train_val_dataset = ReplicatedDataset(train_val_dataset, n_replicas=n_eval_samples)
            log.info(f"   Val dataset: {len(val_dataset)} total samples ({len(val_dataset) // n_eval_samples} unique × {n_eval_samples} replicas)")
            log.info(f"   Test dataset: {len(test_dataset)} total samples ({len(test_dataset) // n_eval_samples} unique × {n_eval_samples} replicas)")
            log.info(f"   Train-Val dataset: {len(train_val_dataset)} total samples ({len(train_val_dataset) // n_eval_samples} unique × {n_eval_samples} replicas)")



        ### Optional: Global normalizer (NOT USED for SP500) ###
        ################### BEGIN ###################
        # NOTE: This global normalizer is DISABLED by default (normalize=False) for SP500.
        # We use per-stock standardization instead (computed in _compute_per_stock_stats).
        # 
        # This code is preserved for users who want to adapt this dataset for other use cases
        # where global normalization across all stocks/timesteps is appropriate.
        # To enable: set dataset.normalize=true in your config.
        # WARNING: Do not use both per-stock standardization AND global normalization!
        
        normalize = cfg.get("normalize", False)
        if normalize: # fit normalizer if enabled
            # Use dataset's cache-specific processed_dir (includes parameter hash)
            actual_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
            normalizer_path = Path(actual_dataset.processed_dir) / "normalizer.json"
            
            if normalizer_path.exists():
                log.info(f"📂 Loading normalizer from {normalizer_path}")
                self.normalizer = Normalizer.load(normalizer_path)
            else:
                log.info("🔧 Fitting normalizer on training data...")
                normalize_method = cfg.get("normalize_method", "standardize")
                normalize_dim = cfg.get("normalize_dim", None)
                # batch_size = cfg.get("batch_size", 32)
                
                self.normalizer = self._fit_normalizer(
                    train_dataset,
                    method=normalize_method,
                    dim=tuple(normalize_dim) if normalize_dim else None,
                    batch_size=32   # Optional: only if using batched data collection
                )
                self.normalizer.save(str(normalizer_path))
                log.info(f"💾 Saved normalizer to {normalizer_path}")
            
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
                log.info(f"   ✅ Normalizer injected into {split_name}")

        ################### END ###################

        # Store reference to train dataset for getting dataset_info later
        self._train_dataset = train_dataset

        # Compute E[σ_cond_w] over the training subset — used to derive the
        # RevIN σ scale correction α(w) automatically via the
        # `${revin_alpha:w,b}` resolver. Logged for visibility in train.log.
        try:
            target_idx = self.feature_names.index(self.target_column_name)
            self._sigma_cond_mean_train = self._compute_sigma_cond_mean(
                train_dataset, target_idx,
            )
            log.info(
                "RevIN σ_cond_w training-set mean = %.4f (target=%s, past_window=%d). "
                "Use ${revin_alpha:w,%.4f} to auto-derive α for any blend weight w.",
                self._sigma_cond_mean_train,
                self.target_column_name,
                past_window,
                self._sigma_cond_mean_train,
            )
        except Exception as e:
            log.warning(
                "Could not compute σ_cond_w training mean (RevIN α auto-derivation "
                "will not be available): %s",
                e,
            )
            self._sigma_cond_mean_train = None

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "train-val": train_val_dataset
        }

    @staticmethod
    def _compute_sigma_cond_mean(
        train_dataset, target_feature_idx: int,
    ) -> float:
        """Compute E[σ_cond_w] over (window, stock) pairs in the train subset.

        For each window in the training subset, extracts the target channel
        from the conditioning tensor ``data.x`` (shape ``[N, T_obs, F_obs]``),
        computes the per-stock std over the T_obs past-window axis, and
        returns the arithmetic mean over all (window, stock) pairs.

        This is the same statistic the diffusion's ``_compute_revin_stats``
        produces at sample-time, aggregated across the train set. Used to
        derive α = 1 / (1 − w·(1 − b)) for the RevIN σ scale correction.
        """
        import torch  # local to keep top-of-module import surface unchanged
        running_sum = 0.0
        running_count = 0
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            # data.x: [N, T_obs, F_obs]
            cond_target = data.x[:, :, target_feature_idx]    # [N, T_obs]
            sigma_w = cond_target.std(dim=1, unbiased=False)  # [N]
            sigma_w = torch.nan_to_num(sigma_w, nan=0.0)
            running_sum += float(sigma_w.sum().item())
            running_count += int(sigma_w.numel())
        if running_count == 0:
            raise RuntimeError("Empty training set; cannot compute σ_cond_w mean.")
        return running_sum / running_count

    def _compute_per_stock_stats(self, values: pd.DataFrame) -> Dict:
        """
        Compute per-stock standardization statistics for SP500.
        
        IMPORTANT: Divides by sp500_scale_factor (100) FIRST to convert
        percentage-scaled values (std~2.0) to raw log returns (std~0.02).
        
        Args:
            values: DataFrame with MultiIndex ['Symbol', 'Date'] and feature columns
        
        Returns:
            Dict mapping feature_name -> {'mean': np.array[N], 'std': np.array[N], 'symbols': list[str]}
            where N is the number of stocks, ordered as they appear in values.csv
        """
        log.info(f"\n🔧 Computing per-stock standardization statistics for SP500...")
        
        # Get unique symbols in the order they appear in the data
        # This MUST match the order used in get_graph_in_pyg_format
        symbols = values.index.get_level_values('Symbol').unique().tolist()
        num_stocks = len(symbols)
        
        # Initialize stats dict with feature-first structure
        stats = {}
        for feat in self.standardized_features:
            stats[feat] = {
                'mean': np.zeros(num_stocks),
                'std': np.zeros(num_stocks),
                'symbols': symbols  # Store symbol ordering for verification
            }
        
        # Reset index to access Symbol column
        values_reset = values.reset_index()
        
        # Compute stats for each stock in order
        for idx, symbol in enumerate(symbols):
            stock_data = values_reset[values_reset['Symbol'] == symbol]
            
            for feat in self.standardized_features:
                if feat in stock_data.columns:
                    # Get raw feature values
                    values_raw = stock_data[feat].values
                    
                    # Apply appropriate preprocessing based on feature type:
                    # 1. Return features (DailyLogReturn, ALR*): divide by 100 to convert from percentage
                    # 2. Volume: log-transform (stats computed on log-scale)
                    # 3. Price features (Open, High, Low), MACD: use as-is
                    
                    if 'Return' in feat or feat.startswith('ALR'):
                        # Divide by scale factor to convert percentage-scaled to raw log returns
                        processed_values = values_raw / self.sp500_scale_factor
                    elif feat in self.log_transform_features:
                        # Log-transform (add 1 to handle zeros: log1p)
                        processed_values = np.log1p(values_raw)
                    else:
                        # Price features, MACD - use as-is
                        processed_values = values_raw
                    
                    # Store mean/std computed on processed values
                    stats[feat]['mean'][idx] = float(processed_values.mean())
                    stats[feat]['std'][idx] = float(processed_values.std())
        
        # Log statistics
        sample_stock_idx = 0
        sample_symbol = symbols[sample_stock_idx]
        sample_mean = stats['DailyLogReturn']['mean'][sample_stock_idx]
        sample_std = stats['DailyLogReturn']['std'][sample_stock_idx]
        log.info(f"   Sample stats ({sample_symbol}): mean={sample_mean:.6f}, std={sample_std:.6f}")
        log.info(f"   Computed stats for {num_stocks} stocks, {len(self.standardized_features)} features")
        log.info(f"   Stock ordering: {symbols[:5]}... (showing first 5)")
        log.info(f"   ✅ Per-stock standardization metadata ready")
        
        return stats
    
    def get_dataset_info(self) -> Optional[Dict]:
        """
        Get dataset standardization metadata for task evaluator.
        
        This is called by trainer to inject standardization info into the task.
        Solution 1: Store metadata at builder level, not in each sample.
        
        Returns:
            Dict with per_stock_stats, feature_names, scale_factor, etc.
        """
        if self.per_stock_stats is None:
            log.info("⚠️  No per_stock_stats computed yet - dataset may not be built")
            return None
        
        return {
            'per_stock_stats': self.per_stock_stats,
            'feature_names': self.feature_names,
            'standardized_features': self.standardized_features,
            'sp500_scale_factor': self.sp500_scale_factor,
            'target_feature_idx': self.feature_names.index(self.target_column_name),  # Use configured target column
            'stock_symbols': self.per_stock_stats[self.target_column_name]['symbols'],  # For error messages

            # Training-set mean of σ_cond_w over (window, stock); used to
            # derive RevIN σ scale correction α = 1/(1−w·(1−b)) via the
            # ${revin_alpha:w,b} OmegaConf resolver. None if not computed.
            'sigma_cond_mean_train': getattr(self, '_sigma_cond_mean_train', None),

            # Backward compatibility with SP100 evaluator
            'log_return_scale_factors': {
                sym: self.per_stock_stats[self.target_column_name]['std'][idx]
                for idx, sym in enumerate(self.per_stock_stats[self.target_column_name]['symbols'])
            }
        }


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
        log.info("📊 Collecting training samples for normalization...")
        log.info(f"   Total samples: {len(train_dataset)}")

        for i in range(len(train_dataset)):
            data = train_dataset[i]
            all_y.append(data.y.cpu())

            if (i + 1) % 100 == 0:
                log.info(f"   Collected {i + 1}/{len(train_dataset)} samples...")
        
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
        #         log.info(f"   Collected {len(all_y)} samples...")


        all_y = torch.cat(all_y, dim=0)
        log.info(f"   Collected {all_y.shape[0]} samples.")
        log.info(f"   Shape: {all_y.shape}")
        log.info(f"   Original - Mean: {all_y.mean():.4f}, Std: {all_y.std():.4f}")
        
        # Fit normalizer
        normalizer = Normalizer(method=method, dim=dim)
        normalizer.fit(all_y)
        
        log.info(f"✅ Normalizer fitted: {normalizer}")
        if method == "standardize":
            log.info(f"   Mean shape: {normalizer.mean.shape}")
            log.info(f"   Std shape: {normalizer.std.shape}")
        
        # Re-enable normalization
        if hasattr(actual_dataset, '_normalize_targets'):
            actual_dataset._normalize_targets = True

        return normalizer
        
    

    def build_loaders(self, cfg: DatasetConfig, datasets: Dict[str, Dataset], accelerator=None) -> Dict[str, DataLoader]:
        # dataset-specific collate can live here if needed
        g = torch.Generator()
        g.manual_seed(42)

        # else:
        log.info("Using standard sequential sampler for training dataloader with shuffling.")
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



        # Get n_samples_per_input for evaluation sampling
        n_samples_per_input = int(cfg.get("n_samples_per_input", 1))
        
        # Calculate originals per batch for evaluation loaders (match WRA strategy)
        # batch_size_eval = originals_per_batch × samples_per_original
        batch_size_val = cfg.get("batch_size_val", cfg.batch_size)  # Use separate eval batch size if specified
        originals_per_batch = math.ceil(batch_size_val / n_samples_per_input)
        eval_batch_size = originals_per_batch * n_samples_per_input
        
        log.info(f"Dataloader configuration:")
        log.info(f"  Training: batch_size={cfg.batch_size}, regular sampling (diversity)")
        log.info(f"  Evaluation: batch_size_val={batch_size_val} → actual batch_size={eval_batch_size}, grouped sampling")
        log.info(f"    (K={n_samples_per_input} samples × {originals_per_batch} timesteps/batch)")
        
        # Validation batch ordering strategy (controlled by shuffle_val config):
        #
        # Both modes always shuffle window order to avoid evaluating on
        # consecutive calendar windows that overlap heavily (e.g. with
        # past_window=20 + future_window=5, adjacent windows share 24/25 days).
        #
        # shuffle_val=True  (default): epoch-varying random order via a stateful
        #     torch.Generator.  The generator advances each epoch, producing a
        #     different permutation every time.  Useful when evaluating on the
        #     full val set so that partial-epoch runs don't always see the same
        #     subset.  Legacy behaviour.
        #
        # shuffle_val=False: epoch-invariant random order via a fixed seed
        #     passed to random.shuffle.  The same permutation is replayed every
        #     epoch, so metrics computed on a small number of val batches
        #     (max_num_val_batches) are directly comparable across epochs and
        #     reference ("_real") statistics stay constant.
        shuffle_val = bool(cfg.get("shuffle_val", True))
        if shuffle_val:
            val_generator = torch.Generator()
            val_generator.manual_seed(43)
            train_val_generator = torch.Generator()
            train_val_generator.manual_seed(44)
            log.info("  shuffle_val=true → val/train-val batches vary across epochs (generator-based)")
        else:
            val_generator = None
            train_val_generator = None
            log.info("  shuffle_val=false → val/train-val batches are randomly selected but fixed across epochs (seed-based)")

        # Create batch samplers for evaluation loaders
        # Use ReplicatedGroupedBatchSampler to guarantee all replicas are batched together
        if n_samples_per_input > 1:
            train_val_sampler = ReplicatedGroupedBatchSampler(
                dataset=datasets["train-val"],
                originals_per_batch=originals_per_batch,
                shuffle=True,
                seed=44 if not shuffle_val else None,
                generator=train_val_generator,
            )
            val_sampler = ReplicatedGroupedBatchSampler(
                dataset=datasets["val"],
                originals_per_batch=originals_per_batch,
                shuffle=True,
                seed=43 if not shuffle_val else None,
                generator=val_generator,
            )
            test_sampler = ReplicatedGroupedBatchSampler(
                dataset=datasets["test"],
                originals_per_batch=originals_per_batch,
                shuffle=False,  # No shuffle for test (deterministic order)
                seed=42  # Fixed seed for test (not used since shuffle=False)
            )
        else:
            # No replication: use None (DataLoader will use default sampler)
            train_val_sampler = None
            val_sampler = None
            test_sampler = None

        # When using batch_sampler, we cannot specify batch_size or shuffle
        # (batch_sampler controls batching entirely)
        if train_val_sampler is not None:
            train_val_loader = DataLoader(
                datasets["train-val"],
                batch_sampler=train_val_sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory, 
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                follow_batch=["y"],
                exclude_keys=["info"]
            )
        else:
            train_val_loader = DataLoader(
                datasets["train-val"],
                batch_size=batch_size_val,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory, 
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                drop_last=cfg.drop_last,
                follow_batch=["y"],
                exclude_keys=["info"]
            )

        if val_sampler is not None:
            val_loader = DataLoader(
                datasets["val"],
                batch_sampler=val_sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                follow_batch=["y"],
                exclude_keys=["info"]
            )
        else:
            val_loader = DataLoader(
                datasets["val"],
                batch_size=batch_size_val,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                drop_last=cfg.drop_last,
                follow_batch=["y"],
                exclude_keys=["info"]
            )

        if test_sampler is not None:
            test_loader = DataLoader(
                datasets["test"],
                batch_sampler=test_sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
                follow_batch=["y"],
                exclude_keys=["info"]
            )
        else:
            test_loader = DataLoader(
                datasets["test"],
                batch_size=batch_size_val,
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
            log.info(f"Plots directory {plots_dir} already exists and is not empty. Skipping plot generation.")
        else:
            log.info(f"Generating plots in {plots_dir} ...")
            # Visualize dataloader evolution for test dataset
            data = next(iter(test_loader))
            stocks_idx = np.random.choice(100, 4)
            num_features = data.x.shape[1]

            target_column_name = test_loader.dataset.dataset.target_column_name if isinstance(test_loader.dataset, (Subset, ReplicatedDataset)) else test_loader.dataset.target_column_name
            info_features = test_loader.dataset.dataset.info['Features'] if isinstance(test_loader.dataset, (Subset, ReplicatedDataset)) else test_loader.dataset.info['Features']
            
            for plot_target in ["Closing_Price", ("y", target_column_name), "Closing_Price_y"] + \
                [(f'x[{i}]', info_features[i]) for i in range(num_features)]:
                plot_batched_data(
                    data = data.clone(),
                    stocks_idx = stocks_idx,
                    plot_target=plot_target,
                    save_dir = f"{cfg.root}/plots",
                )

        return {"train": train_loader, "val": val_loader, "test": test_loader, "train-val": train_val_loader}
