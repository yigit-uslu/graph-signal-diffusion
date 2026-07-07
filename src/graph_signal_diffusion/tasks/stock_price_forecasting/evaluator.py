from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import numpy as np
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.tasks import TASK_REGISTRY
from graph_signal_diffusion.tasks.base import BaseTask
from graph_signal_diffusion.evaluation.metrics import compute_all_metrics
from graph_signal_diffusion.evaluation.probabilistic_metrics import compute_probabilistic_metrics
from graph_signal_diffusion.datasets.normalizer import Normalizer
from graph_signal_diffusion.datasets.sp500.dataset import StocksDataDiffusion
import matplotlib.pyplot as plt
import os
import random

import logging
# logger = logging.getLogger("graph_signal_diffusion.stock_price_forecasting")
logger = logging.getLogger(__name__)


@TASK_REGISTRY.register("stock_price_forecasting")
class StockPriceForecastingTask(BaseTask):
    def __init__(
            self,
            forecast_horizon: int = 5,
            normalizer: Optional[Normalizer] = None,
            n_samples_per_input: int = 20, # 10
            **kwargs
        ):
        super().__init__(**kwargs)
        self.forecast_horizon = forecast_horizon
        self.normalizer = normalizer # NEW normalization support
        self.n_samples_per_input = n_samples_per_input  # Number of samples to generate per input for ensemble predictions
        self.valid_datasets = kwargs.get('valid_datasets', ['sp100', 'sp500'])


    # def set_normalizer(self, normalizer: Normalizer) -> None:
    #     """Set the normalizer for the task."""
    #     self.normalizer = normalizer
    #     print(f"✅ Normalizer injected into {self.__class__.__name__}")


    def prepare_data(self, data_batch: Data) -> Dict[str, Any]:
        """
        Extract and DENORMALIZE stock data from SP100Stocks batch for evaluation.
        
        Args:
            data_batch: PyG Data object with attributes:
                - y: [B*N, T, F] target prices (flattened) or [B, T, N, F]
                - close_price: [B, N, T, 1] or [B, N, F] historical close prices
                - edge_index: [2, E] graph structure
                - edge_weight: [E] edge weights (optional)
                - stocks_index: [B*N] stock identifiers
                - timestamp: [B] time indices
                - info: dict with additional info (e.g., log_return_scale_factors)
        
        Returns:
            Dictionary with 'samples' and 'metadata'
        """

        logger.info(f"Preparing data for stock price forecasting task.")
        logger.info(f"Data batch keys: {data_batch.keys}")

        # 'info' is optional: every downstream use (target-column selection,
        # de-scaling, per-graph metadata) already guards with
        # `hasattr(data_batch, 'info') and data_batch.info is not None` and falls
        # back to defaults. Warn rather than hard-assert so metric-only / mock
        # evaluations without dataset metadata still run.
        if not hasattr(data_batch, 'info') or getattr(data_batch, 'info', None) is None:
            logger.warning(
                "data_batch has no 'info' attribute; proceeding with defaults "
                "(target_column_idx=1, no scale-factor de-scaling)."
            )
        
        # Get target samples (NORMALIZED from the dataset)
        samples = data_batch.y  # Should be [B*N, T, F] or [B, T, N, F]

        #### NEW: DENORMALIZE samples for evaluation if normalizer is set. ####
        if self.normalizer is not None and self.normalizer.fitted:
            logger.info(f"Denormalizing samples for evaluation...")
            samples = self.normalizer.denormalize(samples)
            logger.info(f"After denorm - Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")
            
            # IMPORTANT: Skip de-scaling if normalizer was used, as normalization invalidates the original scale factors
            # The normalizer already brings values back to their original (raw) scale
            logger.info(f"⚠️  Normalizer was used - skipping scale factor de-scaling (already at raw scale)")
            skip_descaling = True
        else:
            logger.info(f"No normalizer set or not fitted; skipping denormalization.")
            skip_descaling = False
        #######################################################################

        # VERIFY if samples are raw log returns using close_price and close_price_y
        if hasattr(data_batch, 'close_price') and hasattr(data_batch, 'close_price_y'):
            logger.info(f"🔍 Verifying target samples are raw log returns...")
            logger.info(f"   samples shape: {samples.shape}, close_price shape: {data_batch.close_price.shape}, close_price_y shape: {data_batch.close_price_y.shape}")
            
            # Get price data
            close_price = data_batch.close_price  # Historical prices
            close_price_y = data_batch.close_price_y  # Future prices
            
            # Compute expected raw log returns from prices
            # For the first future timestep: log(close_price_y[:, 0] / close_price[:, -1])
            if close_price.dim() == 3:  # [B*N, T_hist, 1]
                last_hist_price = close_price[:, -1, :]  # [B*N, 1]
                first_future_price = close_price_y[:, 0, :] if close_price_y.dim() == 3 else close_price_y[:, :, 0]  # [B*N, 1]
                expected_first_lr = torch.log(first_future_price / last_hist_price)  # [B*N, 1]
                
                # Compare with samples (before any de-scaling)
                actual_first_lr = samples[:, 0, :] if samples.dim() == 3 else samples[:, 0]  # [B*N, 1] or similar
                
                # Check if they match (within tolerance for numerical precision)
                abs_diff = (expected_first_lr - actual_first_lr).abs().mean().item()
                rel_diff = (abs_diff / expected_first_lr.abs().mean().item()) if expected_first_lr.abs().mean().item() > 1e-10 else float('inf')
                
                logger.info(f"   Expected first log return (from prices): mean={expected_first_lr.mean().item():.6f}, std={expected_first_lr.std().item():.6f}")
                logger.info(f"   Actual first log return (from samples): mean={actual_first_lr.mean().item():.6f}, std={actual_first_lr.std().item():.6f}")
                logger.info(f"   Abs difference: {abs_diff:.6f}, Rel difference: {rel_diff:.2%}")
                
                if abs_diff < 1e-4:
                    logger.info(f"   ✅ Samples appear to be RAW log returns (no de-scaling needed)")
                    skip_descaling = True
                elif abs_diff < 0.01:
                    logger.info(f"   ⚠️  Small difference detected - samples might need de-scaling")
                else:
                    logger.info(f"   ❌ Large difference - samples are SCALED and need de-scaling")
                
            elif close_price.dim() == 4:  # [B, T_hist, N, 1]
                last_hist_price = close_price[:, -1, :, :]  # [B, N, 1]
                first_future_price = close_price_y[:, 0, :, :] if close_price_y.dim() == 4 else close_price_y  # [B, N, 1]
                expected_first_lr = torch.log(first_future_price / last_hist_price)  # [B, N, 1]
                
                # Compare with samples (assuming [B, T, N, F] after any reshaping)
                if samples.dim() == 3:  # [B*N, T, F] - need to reshape
                    # Will verify after reshaping below
                    expected_first_lr_flat = expected_first_lr.reshape(-1, 1)  # [B*N, 1]
                    actual_first_lr = samples[:, 0, :]  # [B*N, F]
                    
                    # Check if they match (within tolerance for numerical precision)
                    abs_diff = (expected_first_lr_flat - actual_first_lr).abs().mean().item()
                    rel_diff = (abs_diff / expected_first_lr_flat.abs().mean().item()) if expected_first_lr_flat.abs().mean().item() > 1e-10 else float('inf')
                    
                    logger.info(f"   Expected first log return (from prices): mean={expected_first_lr_flat.mean().item():.6f}, std={expected_first_lr_flat.std().item():.6f}")
                    logger.info(f"   Actual first log return (from samples): mean={actual_first_lr.mean().item():.6f}, std={actual_first_lr.std().item():.6f}")
                    logger.info(f"   Abs difference: {abs_diff:.6f}, Rel difference: {rel_diff:.2%}")
                    
                    if abs_diff < 1e-4:
                        logger.info(f"   ✅ Samples appear to be RAW log returns (no de-scaling needed)")
                        skip_descaling = True
                    elif abs_diff < 0.01:
                        logger.info(f"   ⚠️  Small difference detected - samples might need de-scaling")
                    else:
                        logger.info(f"   ❌ Large difference - samples are SCALED and need de-scaling")
                        
                elif samples.dim() == 4:  # [B, T, N, F]
                    actual_first_lr = samples[:, 0, :, :]  # [B, N, F]
                    
                    abs_diff = (expected_first_lr - actual_first_lr).abs().mean().item()
                    rel_diff = (abs_diff / expected_first_lr.abs().mean().item()) if expected_first_lr.abs().mean().item() > 1e-10 else float('inf')
                    
                    logger.info(f"   Expected first log return (from prices): mean={expected_first_lr.mean().item():.6f}, std={expected_first_lr.std().item():.6f}")
                    logger.info(f"   Actual first log return (from samples): mean={actual_first_lr.mean().item():.6f}, std={actual_first_lr.std().item():.6f}")
                    logger.info(f"   Abs difference: {abs_diff:.6f}, Rel difference: {rel_diff:.2%}")
                    
                    if abs_diff < 1e-4:
                        logger.info(f"   ✅ Samples appear to be RAW log returns (no de-scaling needed)")
                        skip_descaling = True
                    elif abs_diff < 0.01:
                        logger.info(f"   ⚠️  Small difference detected - samples might need de-scaling")
                    else:
                        logger.info(f"   ❌ Large difference - samples are SCALED and need de-scaling")

    

        # Get past observations if needed
        past_samples = None
        if hasattr(data_batch, 'x') and data_batch.x is not None:
            past_observations = data_batch.x # should be [B*N, T_obs, F_obs] or [B, T_obs, N, F_obs]

            target_column_idx = data_batch.info.get("Target_column_idx", 1) if hasattr(data_batch, 'info') else 1 
            logger.info("Target column index used for observations extraction: {}/{} of observation features".format(target_column_idx, past_observations.size(-1)))
            past_samples = past_observations[..., target_column_idx:target_column_idx+1] if target_column_idx is not None else past_observations

     
        # Prepare metadata
        metadata = {
            "edge_index": data_batch.edge_index if hasattr(data_batch, 'edge_index') else None,
            "edge_weight": data_batch.edge_weight if hasattr(data_batch, 'edge_weight') else None,
            "stocks_index": data_batch.stocks_index if hasattr(data_batch, 'stocks_index') else None,
            "timestamp": data_batch.timestamp if hasattr(data_batch, 'timestamp') else None
        }
    
        
        # Store batch size and number of stocks for reshaping
        if samples.dim() == 3:  # [B*N, T, F]
            B_N, T, F = samples.shape
            logger.debug(f"Reshape logic might be erroneous for samples shape: {samples.shape}")

            # Infer B and N from stocks_index if available
            if hasattr(data_batch, 'batch'):
                B = data_batch.batch.max().item() + 1
                N = B_N // B
                metadata["batch_size"] = B
                metadata["num_stocks"] = N
                metadata["num_timesteps"] = T
                metadata["num_features"] = F
            
            samples = samples.view(B, N, T, F).permute(0, 2, 1, 3)  # [B, T, N, F]

            if past_samples is not None:
                T_past = past_samples.shape[-2]
                past_samples = past_samples.view(B, N, T_past, -1).permute(0, 2, 1, 3)  # [B, T_obs, N, 1]

            metadata["hist_log_returns"] = past_samples

        elif samples.dim() == 4:  # [B, T, N, F]
            B, T, N, F = samples.shape
            metadata["batch_size"] = B
            metadata["num_stocks"] = N
            metadata["num_timesteps"] = T
            metadata["num_features"] = F

            metadata["hist_log_returns"] = past_samples



        # De-scale target and past samples (reverse pre-processing scale factors from raw CSV)
        # NOTE: Only apply if no normalizer was used (normalizer already handles scale)
        if not skip_descaling and hasattr(data_batch, 'info') and data_batch.info is not None:
            log_return_scale_factors = data_batch.info.get("log_return_scale_factors")
            if log_return_scale_factors is not None:
                from graph_signal_diffusion.datasets.sp100.utils import descale_log_returns
                
                stock_symbols = sorted(log_return_scale_factors.keys())
                # B = data_batch.batch.max().item() + 1
                num_stocks = samples.shape[0] // B if samples.dim() == 3 else samples.shape[-2]
                # num_stocks = samples.shape[0]
                
                if len(stock_symbols) >= num_stocks:
                    stock_symbols = stock_symbols[:num_stocks]
                    logger.info(f"📊 De-scaling {samples.shape} target samples (future log returns)...")
                    
                    # Determine stock dimension based on shape
                    stock_dim = 0 if samples.dim() == 3 else -2
                    samples_before = samples.clone()
                    samples = descale_log_returns(
                        samples, log_return_scale_factors, stock_symbols, stock_dim=stock_dim
                    )
                    logger.info(f"   After de-scaling target samples: std={samples.std().item():.6f}")

                    if past_samples is not None:
                        logger.info(f"📊 De-scaling {past_samples.shape} past samples (historical log returns)...")
                        stock_dim_past = 0 if past_samples.dim() == 3 else -2
                        past_samples = descale_log_returns(
                            past_samples, log_return_scale_factors, stock_symbols, stock_dim=stock_dim_past
                        )
                        logger.info(f"   After de-scaling historical (past observation) samples (log returns): std={past_samples.std().item():.6f}")
                        metadata["hist_log_returns"] = past_samples

                else:
                    logger.warning(f"   ⚠️  Not enough scale factors ({len(stock_symbols)}) for number of stocks ({num_stocks}) - skipping de-scaling")


                # VERIFY AGAIN after de-scaling
                if hasattr(data_batch, 'close_price') and hasattr(data_batch, 'close_price_y'):
                    if samples.dim() == 3:
                        actual_first_lr_after = samples[:, 0, :]
                        abs_diff_after = (expected_first_lr_flat - actual_first_lr_after).abs().mean().item()
                        logger.info(f"   After de-scaling abs difference: {abs_diff_after:.6f}")
                        if abs_diff_after < abs_diff:
                            logger.info(f"   ✅ De-scaling improved match with raw log returns")
                        else:
                            logger.warning(f"   ⚠️  De-scaling made match WORSE - scale factors may be incorrect")
                    elif samples.dim() == 4:
                        actual_first_lr_after = samples[:, 0, :, :]
                        actual_first_lr_after = actual_first_lr_after.reshape(-1, samples.size(-1))
                        abs_diff_after = (expected_first_lr - actual_first_lr_after).abs().mean().item()
                        logger.info(f"   After de-scaling abs difference: {abs_diff_after:.6f}")
                        if abs_diff_after < abs_diff:
                            logger.info(f"   ✅ De-scaling improved match with raw log returns")
                        else:
                            logger.warning(f"   ⚠️  De-scaling made match WORSE - scale factors may be incorrect")


        # # De-scale past_samples (historical log returns) if they're pre-scaled
        # # NOTE: Only apply if no normalizer was used (normalizer already handles scale)
        # if not skip_descaling and hasattr(data_batch, 'info') and data_batch.info is not None:
        #     log_return_scale_factors = data_batch.info.get("log_return_scale_factors")
        #     if log_return_scale_factors is not None:
        #         from graph_signal_diffusion.datasets.sp100.utils import descale_log_returns
                
        #         stock_symbols = sorted(log_return_scale_factors.keys())
        #         num_stocks = past_samples.shape[0] if past_samples.dim() == 3 else past_samples.shape[-2]
                
        #         if len(stock_symbols) >= num_stocks:
        #             stock_symbols = stock_symbols[:num_stocks]
        #             logger.info(f"De-scaling {past_samples.shape} past_samples (historical log returns)...")
                    
        #             # Determine stock dimension based on shape
        #             stock_dim = 0 if past_samples.dim() == 3 else -2
        #             past_samples = descale_log_returns(
        #                 past_samples, log_return_scale_factors, stock_symbols, stock_dim=stock_dim
        #             )
        #             logger.info(f"After de-scaling past_samples: std={past_samples.std().item():.4f}")



        # Extract previous (past) prices for log-return-to-price conversion
        if hasattr(data_batch, 'close_price'):
            # close_price might be [B*N, T, 1] or [B, T, N, 1]
            hist_prices = data_batch.close_price
            
            # Take the last time step as "last observed price" for conversion
            if hist_prices.dim() == 4:  # [B, T, N, 1]
                last_prices = hist_prices[:, -1, :, :]  # [B, N, 1]
                # Store full history for visualization [B, T, N, 1]
                metadata["hist_prices"] = hist_prices

            elif hist_prices.dim() == 3:  # [B*N, T, 1]
                assert samples.size(0) == metadata["batch_size"], \
                    f"Batch size mismatch between samples and close_price: {samples.size(0)} vs {metadata['batch_size']}"
                last_prices = hist_prices[:, -1, :].view(samples.size(0), -1, hist_prices.size(-1))  # [B, N, 1]
                # Reshape history to [B, T, N, 1] for visualization
                B = samples.size(0)
                N = hist_prices.size(0) // B
                T_hist = hist_prices.size(1)
                F_hist = hist_prices.size(2)
                metadata["hist_prices"] = hist_prices.view(B, N, T_hist, F_hist).permute(0, 2, 1, 3)  # [B, T, N, 1]
            
            metadata["last_prices"] = last_prices
            logger.info(f"Extracted metadata[last_prices] shape: {last_prices.shape}")
            logger.info(f"Extracted metadata[hist_prices] shape: {metadata['hist_prices'].shape}")
            
            # Also store for direction accuracy (kept for backward compatibility)
            metadata["prev_prices"] = last_prices


        if hasattr(data_batch, 'close_price_y'):
            # close_price_y might be [B*N, T, 1] or [B, T, N, 1]
            prev_prices_y = data_batch.close_price_y

            # Optionally, denormalize closing prices too
            if self.normalizer is not None and self.normalizer.fitted:
                logger.info(f"Denormalizing close_price_y for metadata...")
                prev_prices_y = self.normalizer.denormalize(prev_prices_y)
                logger.info(f"After denorm - close_price_y Mean: {prev_prices_y.mean():.4f}")
            
            # Take the last time step as "previous future price"
            if prev_prices_y.dim() == 4:  # [B, T, N, 1]
                prev_prices_y = prev_prices_y[:, -1, :, :].unsqueeze(1)  # [B, 1, N, 1]
                
            elif prev_prices_y.dim() == 3:  # [B*N, T, 1]
                assert samples.size(0) == metadata["batch_size"], \
                    f"Batch size mismatch between samples and close_price_y: {samples.size(0)} vs {metadata['batch_size']}"
                prev_prices_y = prev_prices_y[:, -1, :].view(samples.size(0), -1, prev_prices_y.size(-1))  # [B, 1, N, 1]
                prev_prices_y = prev_prices_y.unsqueeze(1)  # [B, 1, N, 1]
            
            metadata["prev_prices_y"] = prev_prices_y
            logger.info(f"Extracted metadata[previous future prices] shape: {prev_prices_y.shape}")

        # Extract log return scale factors and stock symbols from info dict for de-scaling
        # data_batch.info should be injected by the trainer before evaluation
        if hasattr(data_batch, 'info') and data_batch.info is not None:
            info = data_batch.info
            logger.info("Using info from data_batch.info")
            
            if "log_return_scale_factors" in info:
                metadata["log_return_scale_factors"] = info["log_return_scale_factors"]
                # Stock symbols are the keys of scale_factors dict (sorted for consistent ordering)
                metadata["stock_symbols"] = sorted(info["log_return_scale_factors"].keys())
                logger.info(f"Extracted {len(metadata['stock_symbols'])} stock scale factors for de-scaling")
        
        return {
            "samples": samples, # always [B, T, N, F]
            "metadata": metadata,
        }

    def _process_ensemble(
        self,
        generated_samples: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Process ensemble predictions: reshape and compute diversity.
        
        Always treats predictions as ensemble with shape [B, n, T, N, F], where n=1 for non-ensemble case.
        
        Args:
            generated_samples: Raw predictions [B*n, T, N, F] or [B, T, N, F] if n=1
            metadata: Dict containing 'n_samples_per_input' (default 1)
        
        Modifies metadata:
            - 'gen_ensemble': Reshaped ensemble [B, n, T, N, F]
            - 'pred_log_returns': Mean predictions in log return space [B, T, N, F] for later use
        
        Returns:
            Dict with 'sample_diversity' when n>1; empty dict for single-sample
            (diversity is an ensemble-only concept).
        """
        n_samples_per_input = metadata.get('n_samples_per_input', 1)
        
        if n_samples_per_input > 1:
            # Multi-sample case: reshape (B*n, T, N, F) -> (B, n, T, N, F)
            from graph_signal_diffusion.utils import reshape_generated_samples
            gen_reshaped = reshape_generated_samples(generated_samples, n_samples_per_input)
            
            # Compute diversity (std across samples in log return space)
            diversity = gen_reshaped.std(dim=1).mean().item()
        else:
            # Single-sample case: add singleton dimension (B, T, N, F) -> (B, 1, T, N, F)
            gen_reshaped = generated_samples.unsqueeze(1)

        # Store ensemble for downstream processing (descaling, price conversion)
        metadata['gen_ensemble'] = gen_reshaped

        # Store mean predictions in log return space for dimension compatibility
        # NOTE: Proper ensemble mean will be computed on PRICES after conversion
        # This is just for shape handling in descaling step
        metadata['pred_log_returns'] = gen_reshaped.mean(dim=1)  # [B, n, T, N, F] -> [B, T, N, F]

        # Diversity is an ensemble-only concept: report it only for n>1. For a
        # single-sample (point) forecast, omit the key entirely rather than
        # emitting a misleading 0.0.
        if n_samples_per_input > 1:
            return {"sample_diversity": diversity}
        return {}
    
    def _descale_predictions(self, metadata: Dict[str, Any]) -> None:
        """
        De-scale generated predictions if scale factors are available.
        
        Args:
            metadata: Dict containing:
                - 'gen_ensemble': [B, n, T, N, F] ensemble predictions
                - 'pred_log_returns': [B, T, N, F] or [B, 1, T, N, F] ensemble mean
                - 'log_return_scale_factors': Optional dict mapping stock symbols to scale factors
                - 'stock_symbols': Optional list of stock symbols
        
        Modifies metadata in-place:
            - Updates 'gen_ensemble' with descaled values
            - Recomputes 'pred_log_returns' from descaled ensemble mean
        
        Note:
            - Real samples (targets) are already descaled in prepare_data()
            - Skip descaling if normalizer was used (already brings to raw scale)
        """
        pred_log_returns = metadata.get('pred_log_returns')
        gen_ensemble = metadata.get('gen_ensemble')
        log_return_scale_factors = metadata.get("log_return_scale_factors")
        stock_symbols = metadata.get("stock_symbols")
        normalizer_used = self.normalizer is not None and self.normalizer.fitted
        
        logger.info(f"🔍 Checking generated log return scales...")
        logger.info(f"   Generated: mean={pred_log_returns.mean().item():.6f}, std={pred_log_returns.std().item():.6f}")
        
        # Skip descaling if normalizer was used
        if normalizer_used:
            logger.info(f"⚠️  Normalizer was used - skipping scale factor de-scaling for generated samples")
            logger.info(f"   Generated samples already at raw scale after denormalization")
            return
        
        # Descale using per-stock scale factors if available
        if log_return_scale_factors is not None and gen_ensemble is not None:
            from graph_signal_diffusion.datasets.sp100.utils import descale_log_returns
            
            # Get stock symbols list - either from metadata or generate from scale factors
            if stock_symbols is None:
                stock_symbols = sorted(log_return_scale_factors.keys())
            
            # gen_ensemble: [B, n, T, N, F]
            B, n, T, N, F = gen_ensemble.shape
            num_stocks = N
            
            if len(stock_symbols) >= num_stocks:
                stock_symbols = stock_symbols[:num_stocks]
                
                logger.info(f"📊 De-scaling ensemble {gen_ensemble.shape} using per-stock scale factors...")
                # Reshape to (B*n, T, N, F) for de-scaling
                gen_ensemble_flat = gen_ensemble.reshape(B * n, T, N, F)
                gen_ensemble_flat = descale_log_returns(
                    gen_ensemble_flat, log_return_scale_factors, stock_symbols, stock_dim=-2
                )
                # Reshape back to (B, n, T, N, F)
                metadata['gen_ensemble'] = gen_ensemble_flat.reshape(B, n, T, N, F)
                
                # Recompute pred_log_returns from descaled ensemble mean
                metadata['pred_log_returns'] = metadata['gen_ensemble'].mean(dim=1)
                
                logger.info(f"   After de-scaling: ensemble_std={metadata['gen_ensemble'].std().item():.6f}")
                logger.info(f"   Pred mean std={metadata['pred_log_returns'].std().item():.6f}")
            else:
                logger.warning(f"   ⚠️  Not enough scale factors ({len(stock_symbols)}) for number of stocks ({num_stocks}) - skipping de-scaling")
        
        elif log_return_scale_factors is None:
            # Fallback: DETECT if log returns appear to be standardized (z-scored)
            # Real daily log returns have std ~0.01-0.03, z-scored data has std ~1.0
            pred_std = pred_log_returns.std().item()
            is_likely_standardized = pred_std > 0.1
            
            if is_likely_standardized:
                logger.warning(f"⚠️  Generated log returns appear STANDARDIZED (pred_std={pred_std:.3f}).")
                logger.warning(f"   No scale factors in metadata. Price conversion may produce unrealistic values.")
                logger.warning(f"   Consider re-processing dataset to include log_return_scale_factors in info dict.")
    
    def _convert_to_prices(self, target_log_returns: torch.Tensor, metadata: Dict[str, Any]) -> None:
        """
        Convert log returns to actual closing prices.
        
        Args:
            target_log_returns: Ground truth log returns [B, T, N, F] or [B*N, T, F]
            metadata: Dict containing:
                - 'gen_ensemble': [B, n, T, N, F] generated log returns
                - 'pred_log_returns': [B, T, N, F] ensemble mean log returns
                - 'last_prices': [B, N, 1] last observed prices
        
        Modifies metadata in-place:
            - Adds 'target_prices': Ground truth prices
            - Adds 'gen_ensemble_prices': Full ensemble prices [B, n, T, N, F]
            - Adds 'gen_ensemble_log_returns': Clamped log returns [B, n, T, N, F]
            - Adds 'pred_prices': Optional ensemble mean prices [B, T, N, F]
            - Updates 'gen_ensemble': Points to gen_ensemble_prices (backward compat)
        
        Note:
            - Clamps log returns to ±0.5 (50%) per timestep to prevent exp() overflow
            - Formula: price_t = last_price × exp(cumsum(log_returns))
        """
        # Always store target_log_returns FIRST, before any early return, so the
        # no-price fallback in _compute_all_metrics can still compute logreturn_*
        # metrics when price conversion is skipped below (missing last_prices).
        metadata['target_log_returns'] = target_log_returns

        last_prices = metadata.get("last_prices")
        if last_prices is None:
            logger.warning("⚠️  No last_prices in metadata. Skipping price conversion.")
            return

        pred_log_returns = metadata.get('pred_log_returns')
        gen_ensemble = metadata.get('gen_ensemble')

        # CLAMP log returns to prevent exp() overflow (realistic daily returns rarely exceed ±50%)
        LOG_RETURN_CLAMP = 0.5  # ±50% max per-step return
        pred_log_returns_clamped = torch.clamp(pred_log_returns, -LOG_RETURN_CLAMP, LOG_RETURN_CLAMP)
        target_log_returns_clamped = torch.clamp(target_log_returns, -LOG_RETURN_CLAMP, LOG_RETURN_CLAMP)
        
        # Check if clamping was needed and warn
        pred_clamped_pct = (pred_log_returns.abs() > LOG_RETURN_CLAMP).float().mean().item() * 100
        target_clamped_pct = (target_log_returns.abs() > LOG_RETURN_CLAMP).float().mean().item() * 100
        if pred_clamped_pct > 1 or target_clamped_pct > 1:
            pred_std = pred_log_returns.std().item()
            target_std = target_log_returns.std().item()
            logger.warning(f"⚠️  Clamped {pred_clamped_pct:.1f}% pred, {target_clamped_pct:.1f}% target log returns to ±{LOG_RETURN_CLAMP}")
            logger.warning(f"   This suggests values are not at raw log return scale (expected std ~0.01-0.03)")
            logger.warning(f"   Current std: pred={pred_std:.4f}, target={target_std:.4f}")
        
        if pred_log_returns.dim() == 4:  # [B, T, N, F]
            # Cumulative sum over time dimension
            cumulative_log_returns_pred = torch.cumsum(pred_log_returns_clamped, dim=1)  # [B, T, N, F]
            cumulative_log_returns_target = torch.cumsum(target_log_returns_clamped, dim=1)  # [B, T, N, F]
            
            # Expand last_prices to match: [B, N, 1] -> [B, 1, N, 1]
            last_prices_expanded = last_prices.unsqueeze(1)  # [B, 1, N, 1]
            
            # Convert to prices
            pred_prices = last_prices_expanded * torch.exp(cumulative_log_returns_pred)
            target_prices = last_prices_expanded * torch.exp(cumulative_log_returns_target)
            
            # Store target prices in metadata for visualization
            metadata['target_prices'] = target_prices
            
            # Also convert ensemble if available
            if gen_ensemble is not None:  # (B, n, T, N, F) - in raw log return space
                logger.info(f"💰 Converting ensemble {gen_ensemble.shape} from log returns to prices...")
                # Clamp ensemble log returns
                gen_ensemble_clamped = torch.clamp(gen_ensemble, -LOG_RETURN_CLAMP, LOG_RETURN_CLAMP)
                
                # Store the clamped log returns in metadata for visualization's cumulative plot
                metadata['gen_ensemble_log_returns'] = gen_ensemble_clamped
                
                # Cumulative sum over time (dim=2)
                cumulative_log_returns_ens = torch.cumsum(gen_ensemble_clamped, dim=2)  # (B, n, T, N, F)
                # Expand last_prices: [B, N, 1] -> [B, 1, 1, N, 1]
                last_prices_ens = last_prices.unsqueeze(1).unsqueeze(1)
                # Convert to prices
                ensemble_prices = last_prices_ens * torch.exp(cumulative_log_returns_ens)
                
                # Store both versions explicitly for clarity
                metadata['gen_ensemble_prices'] = ensemble_prices
                metadata['gen_ensemble'] = ensemble_prices  # Keep for backward compatibility
                logger.info(f"   Ensemble prices: mean={ensemble_prices.mean().item():.2f}, std={ensemble_prices.std().item():.2f}")
                
                # Compute ensemble mean on PRICES (correct way)
                # mean(exp(cumsum(log_returns))) is the proper ensemble mean for price predictions
                pred_prices = ensemble_prices.mean(dim=1)  # (B, n, T, N, F) -> (B, T, N, F)
                metadata['pred_prices'] = pred_prices
                logger.info(f"   Using ensemble mean computed on PRICES for primary metrics")
        
        else:  # [B*N, T, F]
            # Flatten last_prices: [B, N, 1] -> [B*N, 1]
            B_N, T, F = pred_log_returns.shape
            last_prices_flat = last_prices.reshape(B_N, 1)  # [B*N, 1]
            
            # Cumulative sum over time (use clamped values)
            cumulative_log_returns_pred = torch.cumsum(pred_log_returns_clamped, dim=1)  # [B*N, T, F]
            cumulative_log_returns_target = torch.cumsum(target_log_returns_clamped, dim=1)  # [B*N, T, F]
            
            # Convert to prices
            pred_prices = last_prices_flat.unsqueeze(-1) * torch.exp(cumulative_log_returns_pred)
            target_prices = last_prices_flat.unsqueeze(-1) * torch.exp(cumulative_log_returns_target)
            
            # Store target prices in metadata for visualization
            metadata['target_prices'] = target_prices
            metadata['pred_prices'] = pred_prices
    
    def _compute_all_metrics(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            metadata: Dict containing:
                - 'pred_prices': [B, T, N, F] predicted prices
                - 'target_prices': [B, T, N, F] ground truth prices
                - 'pred_log_returns': [B, T, N, F] predicted log returns
                - 'target_log_returns': [B, T, N, F] ground truth log returns
                - 'gen_ensemble': [B, n, T, N, F] ensemble predictions (optional)
        
        Returns:
            Dictionary of all computed metrics (primary, probabilistic, stock-specific, temporal, return)
        
        Note:
            Does not include diversity_metrics - those should be merged separately
        """
        pred_prices = metadata.get('pred_prices')
        target_prices = metadata.get('target_prices')
        pred_log_returns = metadata.get('pred_log_returns')
        target_log_returns = metadata.get('target_log_returns')
        
        # Compute metrics
        if pred_prices is not None and target_prices is not None:
            # PRIMARY METRICS: Compute on actual closing prices
            metrics = compute_all_metrics(pred_prices, target_prices, prefix="price_")
            
            # Compute probabilistic metrics on PRICES if ensemble is available
            gen_ensemble_prices = metadata.get('gen_ensemble', None)
            # Probabilistic metrics (CRPS, coverage, ...) are ensemble-only: require
            # a real ensemble dimension (n>1). For single-sample (n=1) forecasts the
            # ensemble axis is a singleton, so skip them. Matches the n>1 guard used
            # for the return-space probabilistic metrics elsewhere in this file.
            if gen_ensemble_prices is not None and gen_ensemble_prices.shape[1] > 1:  # (B, n, T, N, F) - now in PRICE space
                logger.info(f"📊 Computing probabilistic metrics on ensemble prices...")
                prob_metrics = compute_probabilistic_metrics(
                    predictions=gen_ensemble_prices,  # (B, n, T, N, F) - PRICES
                    target=target_prices  # (B, T, N, F) - PRICES
                )
                metrics.update(prob_metrics)
                logger.info(f"   Probabilistic metrics computed on actual closing prices")
            
            # Stock-specific metrics on prices
            stock_metrics = self._compute_stock_specific_metrics(
                pred_prices, target_prices, metadata
            )
            metrics.update(stock_metrics)
            
            # Temporal metrics on prices (error at different horizons)
            if pred_prices.dim() == 4 and pred_prices.size(1) > 1:  # [B, T, N, F] with T > 1
                temporal_metrics = self._compute_temporal_metrics(pred_prices, target_prices, prefix="price_")
                metrics.update(temporal_metrics)
        
        else:
            # Fallback: No price conversion possible, compute metrics on log returns directly
            logger.warning("⚠️  No prices in metadata. Computing metrics on log returns directly.")
            metrics = compute_all_metrics(pred_log_returns, target_log_returns, prefix="logreturn_")
            stock_metrics = self._compute_stock_specific_metrics(
                pred_log_returns, target_log_returns, metadata
            )
            metrics.update(stock_metrics)
        
        # SECONDARY METRICS: Diagnostics on log returns
        return_metrics = self._compute_return_metrics(pred_log_returns, target_log_returns)
        metrics.update(return_metrics)
        
        return metrics
    
    def _visualize_results(self, metadata: Dict[str, Any], viz_save_dir: Optional[str] = None) -> None:
        """
        Visualize prediction results for several batches.
        
        Args:
            metadata: Dict containing predictions, targets, last (past) prices etc.
            viz_save_dir: Optional directory to save visualizations
        
        Note:
            Visualizes first 4 batches to save time
        """
        # No save directory -> nothing to persist. Skip figure construction
        # entirely (the figures would be built and immediately closed/discarded),
        # which also avoids exercising viz-only code paths during metric-only eval.
        if viz_save_dir is None:
            return

        pred_log_returns = metadata.get('pred_log_returns')
        target_log_returns = metadata.get('target_log_returns')
        
        logger.info(f"📊 Preparing visualization:")
        logger.info(f"   pred_log_returns shape: {pred_log_returns.shape}")
        logger.info(f"   pred_log_returns stats: mean={pred_log_returns.mean().item():.6f}, std={pred_log_returns.std().item():.6f}")
        logger.info(f"   target_log_returns stats: mean={target_log_returns.mean().item():.6f}, std={target_log_returns.std().item():.6f}")

        # Visualize stock predictions for several batches
        # Determine batch dimension - handle both [B, T, N, F] and [B, 1, T, N, F]
        B = pred_log_returns.size(0)
        for idx in range(min(4, B)):  # Visualize only first four samples in the batch to save time
            stocks = None # [0, 10, 20, 30, 40]
            n_stock_examples = len(stocks) if stocks is not None else 5
            fig = self.visualize_predictions(
                metadata=metadata,
                stocks=stocks,
                n_examples=n_stock_examples,
                figsize=(12, 6 * n_stock_examples),
                batch_index=idx,
                save_dir=viz_save_dir,
                rng_seed=int(42 + idx)
            )

            plt.close(fig)  # Close figure to free memory after saving
    
    def evaluate_samples(
        self, 
        generated_samples: torch.Tensor,
        real_samples: torch.Tensor,
        metadata: Dict[str, Any],
        viz_save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate stock price forecasting.
        
        Args:
            generated_samples: Generated predictions [B, T, N, F] or [B*N, T, F] or [B*n, T, N, F] when n_samples_per_input > 1
            real_samples: Ground truth [B, T, N, F] or [B*N, T, F]
            metadata: Dictionary containing edge_index, prev_prices, etc. May include 'n_samples_per_input' if > 1
            viz_save_dir: Optional directory to save visualizations. By default, saves to Hydra output dir if active.
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Process ensemble predictions
        diversity_metrics = self._process_ensemble(generated_samples, metadata)
        
        # De-scale predictions if scale factors are available
        self._descale_predictions(metadata)
        
        # Get processed predictions from metadata (now descaled)
        pred_log_returns = metadata.get('pred_log_returns')
        hist_log_returns = metadata.get('hist_log_returns')
        target_log_returns = real_samples
        
        logger.info(f"🔍 Checking target log return scales...")
        logger.info(f"   Target: mean={target_log_returns.mean().item():.6f}, std={target_log_returns.std().item():.6f}")

        logger.info(f"🔍 Checking historical log return scales...")
        if hist_log_returns is not None:
            logger.info(f"   Historical: mean={hist_log_returns.mean().item():.6f}, std={hist_log_returns.std().item():.6f}")
        else:
            logger.info(f"   Historical: not available (prepare_data was not called)")


        # Convert log returns to actual closing prices
        self._convert_to_prices(target_log_returns, metadata)
        
        # Compute all metrics
        metrics = self._compute_all_metrics(metadata)
        
        # Add diversity metrics if multiple samples were generated
        metrics.update(diversity_metrics)
        
        # Visualize results
        self._visualize_results(metadata, viz_save_dir)
        
        return metrics
    

    def _plot_time_series(
            self,
            ax: plt.Axes,
            default_time_idx,
            series_dict,
            ylabel: str,
            title: Optional[str] = None,
            marker: Optional[Union[str, dict]] = None,
            linecolor: Optional[Union[str, dict]] = None,
            linestyle: Optional[Union[str, dict]] = "-",
            legend: bool = True,
        ) -> plt.Axes:
        """Helper to plot multiple series on a single axis.

        Each entry in series_dict may be either:
          - a 1D array-like (same length as default_time_idx), or
          - a (time_idx, series) tuple where time_idx and series are array-like

        Args:
            ax: matplotlib axis
            default_time_idx: sequence of x values to use if a series doesn't provide its own x
            series_dict: dict[label] = series or (x, series)
            ylabel: y-axis label
            title: optional title
            marker: optional marker for all series or a dictionary of label->marker with labels matching series_dict keys
            linecolor: optional color for all series or a dictionary of label->color with labels matching series_dict keys
            linestyle: optional linestyle for all series or a dictionary of label->linestyle with labels matching series_dict keys
            legend: whether to show legend
        """

        for label, series in series_dict.items():
            # Unpack per-series x if provided
            if isinstance(series, (list, tuple)) and len(series) == 2:
                x_vals, y_vals = series
            else:
                x_vals, y_vals = default_time_idx, series

            # Accept torch tensors or numpy arrays
            if hasattr(y_vals, 'cpu'):
                y_vals = y_vals.cpu().numpy()
            if hasattr(x_vals, 'cpu'):
                x_vals = x_vals.cpu().numpy()

            # Determine marker
            if isinstance(marker, dict):
                series_marker = marker.get(label, None)
            else:
                series_marker = marker

            # Determine linecolor
            if isinstance(linecolor, dict):
                series_color = linecolor.get(label, None)
            else:
                series_color = linecolor

            # Determine linestyle
            if isinstance(linestyle, dict):
                series_linestyle = linestyle.get(label, "-")
            else:
                series_linestyle = linestyle
            
            # Use transparency for ensemble samples
            is_sample = label.startswith('Sample ')
            alpha = 0.5 if is_sample else 1.0
            linewidth = 1.0 if is_sample else 2.0
            plot_label = None if is_sample else label  # Don't show individual sample labels

            ax.plot(x_vals, y_vals, label=plot_label,
                    color = series_color,
                    marker=series_marker,
                    linewidth=linewidth,
                    linestyle=series_linestyle,
                    alpha=alpha
                    )
        if title:
            ax.set_title(title)
        
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.1)
        if series_dict and legend:
            ax.legend(loc='upper left')

        return ax

    def visualize_predictions(
        self,
        metadata: Dict[str, Any],
        stocks: Optional[list[int]] = None,
        batch_index: int = 0,
        n_examples: int = 5,
        figsize: tuple = (12, 8),
        save_dir: Optional[str] = None,
        plot_cumulative: bool = True,
        show_confidence_bands: bool = True,
        confidence_alpha: float = 0.1,
        rng_seed: int = 42,
    ) -> plt.Figure:
        """Create overlay plots of predicted vs actual stock prices for a subset of stocks.

        Args:
            metadata: Metadata dict containing:
                - 'gen_ensemble' or 'gen_ensemble_prices': ensemble predictions as PRICES [B, n, T, N, F]
                - 'gen_ensemble_log_returns': ensemble as raw log returns for cumulative plot [B, n, T, N, F]
                - 'target_prices': ground truth prices [B, T, N, F]
                - 'hist_log_returns': historical raw daily log returns for cumulative plot
                - 'hist_prices': historical prices for price plot
                - 'last_prices': last observed prices
            stocks: Optional list of stock indices to plot. If None, randomly sample `n_examples` stocks using the provided rng seed.
            batch_index: Which batch sample to use for plotting (default 0)
            n_examples: Number of stocks to plot when `stocks` is None
            figsize: Figure size
            save_dir: If provided, save the figure to this directory. If not, and Hydra is active, save to Hydra output dir.
            plot_cumulative: If True, show both prices and cumulative log returns.
            show_confidence_bands: If True and ensemble available, draw shaded confidence bands (default: True)
            confidence_alpha: Significance level for confidence bands, e.g., 0.1 for 90% interval (default: 0.1)
            rng_seed: Random seed for stock sampling when `stocks` is None.

        Returns:
            Matplotlib Figure object.
        """
        
        # Get data from metadata
        ensemble_log_returns_orig = metadata.get('gen_ensemble_log_returns', None)
        assert ensemble_log_returns_orig is not None, \
            logger.warning("Metadata must contain 'gen_ensemble_log_returns' for cumulative plot but has keys: {}".format(list(metadata.keys())))
        
        # Get ensemble prices - check explicit naming first, then fall back to gen_ensemble
        gen_ensemble = metadata.get('gen_ensemble_prices', None)
        if gen_ensemble is None:
            gen_ensemble = metadata.get('gen_ensemble', None)  # (B, n, T, N, F) prices
        
        # Get predictions: compute from ensemble mean or use stored pred_prices
        if gen_ensemble is not None:
            # Compute ensemble mean in price space
            pred = gen_ensemble.mean(dim=1)  # [B, n, T, N, F] -> [B, T, N, F]
        elif 'pred_prices' in metadata:
            pred = metadata['pred_prices']
        else:
            raise ValueError("metadata must contain gen_ensemble/gen_ensemble_prices or pred_prices")
        
        # Get targets from metadata
        if 'target_prices' not in metadata:
            raise ValueError("metadata must contain target_prices")
        target = metadata['target_prices']
        
        # Retrieve historical log returns from metadata (these are RAW daily log returns for cumulative plot)
        hist_log_returns = metadata.get("hist_log_returns", None)
        if hist_log_returns is None:
            logger.warning("No historical log returns in metadata for cumulative plot. Metadata has keys: {}".format(list(metadata.keys())))
        else:
            assert hist_log_returns.shape[0] == target.shape[0], \
                f"Batch size mismatch between hist_log_returns and target: {hist_log_returns.shape[0]} vs {target.shape[0]}"
            
            assert hist_log_returns.dim() == target.dim(), \
                f"Dimensionality mismatch between hist_log_returns and target: {hist_log_returns.dim()} vs {target.dim()}"
        
        # Retrieve historical closing prices (ACTUAL PRICES for visualization)
        hist_prices = metadata.get("hist_prices", None)  # [B, T_hist, N, 1]
        if hist_prices is not None:
            assert hist_prices.shape[0] == target.shape[0], \
                f"Batch size mismatch between hist_prices and target: {hist_prices.shape[0]} vs {target.shape[0]}"
        
        # Get last observed price for connecting history to forecast in price plot
        last_prices = metadata.get("last_prices", None)  # [B, N, 1]
        assert last_prices is not None, "Metadata must contain last_prices for price forecasts but has keys: {}".format(list(metadata.keys()))
        last_price_val = last_prices  # This is the starting point for price forecasts
        

        # Ensure inputs are in [B, T, N, F] format
        if pred.dim() == 3:
            # [B*N, T, F] -> try to reshape using metadata info
            B = metadata.get('batch_size', 1)
            N = metadata.get('num_stocks', pred.size(0) // B)
            pred = pred.view(B, N, pred.size(1), pred.size(2)).permute(0, 2, 1, 3)  # [B, T, N, F := 1]
            target = target.view(B, N, target.size(1), target.size(2)).permute(0, 2, 1, 3)  # [B, T, N, F := 1]
            if hist_log_returns is not None:
                hist_log_returns = hist_log_returns.view(B, N, hist_log_returns.size(1), hist_log_returns.size(2)).permute(0, 2, 1, 3)  # [B, T_obs, N, F_obs := 1]
            # hist_prices is already in [B, T, N, F] format from prepare_data
        elif pred.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {pred.shape}")

        B, T, N, F = pred.shape
        T_hist = hist_log_returns.size(1) if hist_log_returns is not None else 0
        # Also check hist_prices for T_hist if hist_log_returns not available
        if T_hist == 0 and hist_prices is not None:
            T_hist = hist_prices.size(1)

        # Select batch sample
        if batch_index >= B:
            raise IndexError(f"batch_index {batch_index} out of range (B={B})")


        pred_b = pred[batch_index]  # (T, N, F) - PRICES
        target_b = target[batch_index]  # (T, N, F) - PRICES
        hist_log_b = hist_log_returns[batch_index] if hist_log_returns is not None else None  # (T_obs, N, F_obs) - LOG RETURNS
        
        # Get historical closing prices for this batch (for price plot)
        hist_prices_b = hist_prices[batch_index] if hist_prices is not None else None  # (T_hist, N, 1) - ACTUAL PRICES
        
        # Get last price for this batch to connect history to forecast
        last_price_b = last_price_val[batch_index] if last_price_val is not None else None  # (N, 1)
        
        # Extract ensemble samples if available
        if gen_ensemble is not None:
            ensemble_b = gen_ensemble[batch_index]  # (n, T, N, F) - PRICES
            n_samples = ensemble_b.shape[0]
            # Also get ensemble log returns for cumulative plot
            if ensemble_log_returns_orig is not None:
                ensemble_log_b = ensemble_log_returns_orig[batch_index]  # (n, T, N, F) - LOG RETURNS
            else:
                ensemble_log_b = None
        else:
            ensemble_b = None
            ensemble_log_b = None
            n_samples = 1
        
        # Convert to 2D time series for plotting (T, N) - use first feature if F>1
        # Price series (already in price space)
        if F > 1:
            pred_ts = pred_b[..., 0]  # (T, N) - PRICES
            target_ts = target_b[..., 0]  # (T, N) - PRICES
            ensemble_ts = ensemble_b[..., 0] if ensemble_b is not None else None  # (n, T, N) - PRICES
            hist_log_ts = hist_log_b[..., 0] if hist_log_b is not None else None  # (T_obs, N) - LOG RETURNS
        else:
            pred_ts = pred_b.squeeze(-1)  # (T, N) - PRICES
            target_ts = target_b.squeeze(-1)  # (T, N) - PRICES
            ensemble_ts = ensemble_b.squeeze(-1) if ensemble_b is not None else None  # (n, T, N) - PRICES
            hist_log_ts = hist_log_b.squeeze(-1) if hist_log_b is not None else None  # (T_obs, N) - LOG RETURNS
        
        # Ensemble log returns for cumulative plot
        if ensemble_log_b is not None:
            if F > 1:
                ensemble_log_ts = ensemble_log_b[..., 0]  # (n, T, N)
            else:
                ensemble_log_ts = ensemble_log_b.squeeze(-1)
        else:
            ensemble_log_ts = None
        
        # Historical prices series (for price plot)
        if hist_prices_b is not None:
            hist_prices_ts = hist_prices_b[..., 0] if hist_prices_b.size(-1) > 1 else hist_prices_b.squeeze(-1)  # (T_hist, N) - ACTUAL PRICES
        else:
            hist_prices_ts = None

        # Choose stocks to plot
        if stocks is None:
            stocks = list(range(N))
            rng = random.Random(rng_seed)  # Fixed seed for reproducibility
            if N > n_examples:
                stocks = rng.sample(stocks, n_examples)
            else:
                stocks = stocks
        else:
            # Validate stocks indices
            stocks = [s for s in stocks if 0 <= s < N]
            if not stocks:
                raise ValueError("No valid stock indices provided for plotting.")

        # Number of plots (one per selected stock)
        n_plots = len(stocks)

        # Setup axes: if plotting cumulative returns, allocate two rows per stock
        if plot_cumulative:
            rows = n_plots * 2
            fig, axes = plt.subplots(rows, 1, figsize=(figsize[0], max(figsize[1], 2 * rows)), sharex=True)
            axes = list(axes)
        else:
            rows = n_plots
            fig, axes = plt.subplots(rows, 1, figsize=(figsize[0], max(figsize[1], 2 * rows)), sharex=True)
            if n_plots == 1:
                axes = [axes]

        ts = metadata.get('timestamp', None)
        if ts is None:
            time_start = 0
        else:
            # timestamps is [B_original * N] = [B * n * N,] tensor where n = n_samples_per_input
            # batch_index iterates over B unique timesteps (after ensemble de-duplication),
            # so we must skip n*N entries per unique timestep (not just N).
            n_spi = metadata.get('n_samples_per_input', 1)
            if n_spi is None:
                n_spi = 1
            temp = ts[batch_index * n_spi * N]
            time_start = temp.item() if hasattr(temp, 'item') else int(temp)

        time_idx = list(range(time_start, time_start + T + T_hist))
        for plot_idx, s in enumerate(stocks):
            # Prepare series for PRICE plot
            # Use last observed price as starting point to connect to forecasts
            if last_price_b is not None:
                last_p = last_price_b[s, 0]  # Scalar: last observed price for stock s
                
                # If ensemble available, compute mean in PRICE space (correct)
                # Note: mean(exp(cumsum(log_returns))) ≠ exp(cumsum(mean(log_returns)))
                if ensemble_ts is not None:
                    pred_mean_price = ensemble_ts[:, :, s].mean(dim=0)  # Mean of prices across samples (T,)
                    logger.info(f"Stock {s}: using ensemble mean prices, first 3 timesteps: {pred_mean_price[:3].cpu().numpy()}")
                else:
                    pred_mean_price = pred_ts[:, s]  # Fallback to converted mean log returns (T,)
                    logger.info(f"Stock {s}: using pred_ts (converted mean log returns), first 3 timesteps: {pred_mean_price[:3].cpu().numpy()}")
                
                # Prepend last price to create continuity
                pred_cts = torch.cat((last_p.unsqueeze(0), pred_mean_price), dim=0)  # (1+T,)
                target_cts = torch.cat((last_p.unsqueeze(0), target_ts[:, s]), dim=0)  # (1+T,)
                
                logger.info(f"Stock {s}: last_p={last_p.item():.2f}, target_cts first 3: {target_cts[:3].cpu().numpy()}, pred_cts first 3: {pred_cts[:3].cpu().numpy()}")
                
                # Time indices for forecast (starting from last observation point)
                forecast_time_idx = list(range(time_start + T_hist - 1, time_start + T_hist + T))
                
                # Time indices for history
                history_time_idx = list(range(time_start, time_start + T_hist))
                
                series_daily = {
                    'Target': (forecast_time_idx, target_cts),
                    'Predicted (mean)': (forecast_time_idx, pred_cts),
                }
                
                # Add full observation history if available (actual closing prices in black)
                if hist_prices_ts is not None:
                    series_daily['Observed History'] = (history_time_idx, hist_prices_ts[:, s])
                else:
                    # Fallback: just show last observed price point
                    series_daily['Last Observed'] = ([time_start + T_hist - 1], [last_p])
                
                default_x = forecast_time_idx
            else:
                # If ensemble available, compute mean in price space
                if ensemble_ts is not None:
                    pred_mean_price = ensemble_ts[:, :, s].mean(dim=0)
                else:
                    pred_mean_price = pred_ts[:, s]
                
                series_daily = {
                    'Target': (time_idx[-T:], target_ts[:, s]),
                    'Predicted (mean)': (time_idx[-T:], pred_mean_price),
                }
                default_x = time_idx[-T:]
            
            # Add ensemble members if available (PRICES)
            if ensemble_ts is not None:
                for sample_idx in range(n_samples):
                    sample_ts_price = ensemble_ts[sample_idx, :, s]  # (T,)
                    if last_price_b is not None:
                        sample_cts = torch.cat((last_p.unsqueeze(0), sample_ts_price), dim=0)
                        series_daily[f'Sample {sample_idx+1}'] = (forecast_time_idx, sample_cts)
                    else:
                        series_daily[f'Sample {sample_idx+1}'] = (time_idx[-T:], sample_ts_price)

            ax_daily = axes[plot_idx * (2 if plot_cumulative else 1)]
            
            # Build plotting kwargs with ensemble members styled differently
            marker = {"Observed History": "o", "Last Observed": "o", "Target": "d", "Predicted (mean)": "x"}
            linecolor = {"Observed History": "black", "Last Observed": "black", "Target": "tab:blue", "Predicted (mean)": "tab:orange"}
            linestyle = {"Observed History": "-", "Last Observed": "", "Target": "-", "Predicted (mean)": "--"}
            
            if ensemble_ts is not None:
                for sample_idx in range(n_samples):
                    marker[f'Sample {sample_idx+1}'] = None
                    linecolor[f'Sample {sample_idx+1}'] = "tab:orange"
                    linestyle[f'Sample {sample_idx+1}'] = "-"
            
            self._plot_time_series(
                ax_daily,
                default_x,
                series_daily,
                ylabel='Closing\nPrice', # 'Closing\nPrice (\$)',
                title=f"Stock id: {s}",
                marker=marker,
                linecolor=linecolor,
                linestyle=linestyle,
                legend=(plot_idx == 0),  # Show legend only on first plot
                )
            
            # Add confidence bands if ensemble is available and requested (on PRICES)
            if ensemble_ts is not None and show_confidence_bands:
                # Compute quantiles for confidence bands
                lower_q = confidence_alpha / 2
                upper_q = 1 - confidence_alpha / 2
                
                # Get ensemble for this stock: (n, T) - PRICES
                ensemble_stock = ensemble_ts[:, :, s]  # (n_samples, T)
                
                # Compute quantiles
                lower_band = torch.quantile(ensemble_stock, lower_q, dim=0)  # (T,)
                upper_band = torch.quantile(ensemble_stock, upper_q, dim=0)  # (T,)
                
                # Add last price for continuity if present
                if last_price_b is not None:
                    lower_cts = torch.cat((last_p.unsqueeze(0), lower_band), dim=0)
                    upper_cts = torch.cat((last_p.unsqueeze(0), upper_band), dim=0)
                    band_x = forecast_time_idx
                else:
                    lower_cts = lower_band
                    upper_cts = upper_band
                    band_x = time_idx[-T:]
                
                # Plot shaded band
                ax_daily.fill_between(
                    band_x,
                    lower_cts.cpu().numpy(),
                    upper_cts.cpu().numpy(),
                    color='tab:orange',
                    alpha=0.2,
                    label=f'{int((1-confidence_alpha)*100)}% CI' if plot_idx == 0 else None
                )
                
                # Update legend if we added the band
                if plot_idx == 0:
                    ax_daily.legend(loc='upper left')

            # Secondary axis: cumulative LOG RETURNS (using historical log return data)
            if plot_cumulative and hist_log_ts is not None:
                ax_cum = axes[plot_idx * 2 + 1]

                # Build cumulative log returns with history shown separately in black
                # Compute cumulative log returns for history
                cum_hist = torch.cumsum(hist_log_ts[:, s], dim=0)  # (T_hist,)
                last_cum_hist = cum_hist[-1]  # Last cumulative value to continue from
                
                # For forecast: derive log returns from prices
                # target log returns: log(target_price[t] / target_price[t-1])
                target_price_series = target_ts[:, s]  # (T,) prices
                if last_price_b is not None:
                    last_p = last_price_b[s, 0]
                    # Prepend last price to compute returns
                    target_with_last = torch.cat([last_p.unsqueeze(0), target_price_series], dim=0)  # (T+1,)
                    target_log_returns_derived = torch.log(target_with_last[1:] / target_with_last[:-1])  # (T,)
                else:
                    # Can't derive without last price
                    target_log_returns_derived = None
                
                if target_log_returns_derived is not None:
                    cum_target_forecast = last_cum_hist + torch.cumsum(target_log_returns_derived, dim=0)  # (T,)
                    
                    # For predicted mean: use ensemble log returns if available
                    if ensemble_log_ts is not None:
                        # Mean of cumulative log returns across samples
                        cum_pred_all = last_cum_hist + torch.cumsum(ensemble_log_ts[:, :, s], dim=1)  # (n, T)
                        cum_pred_mean = cum_pred_all.mean(dim=0)  # (T,)
                    else:
                        # Derive from pred prices
                        pred_price_series = pred_ts[:, s]  # (T,) prices
                        pred_with_last = torch.cat([last_p.unsqueeze(0), pred_price_series], dim=0)  # (T+1,)
                        pred_log_returns_derived = torch.log(pred_with_last[1:] / pred_with_last[:-1])  # (T,)
                        cum_pred_mean = last_cum_hist + torch.cumsum(pred_log_returns_derived, dim=0)  # (T,)
                    
                    # Prepend last history point for continuity
                    cum_target_cts = torch.cat((last_cum_hist.unsqueeze(0), cum_target_forecast), dim=0)  # (1+T,)
                    cum_pred_cts = torch.cat((last_cum_hist.unsqueeze(0), cum_pred_mean), dim=0)  # (1+T,)
                    
                    # Time indices
                    history_time_idx_cum = list(range(time_start, time_start + T_hist))
                    forecast_time_idx_cum = list(range(time_start + T_hist - 1, time_start + T_hist + T))
                    
                    series_cum = {
                        'Observed History': (history_time_idx_cum, cum_hist),
                        'Target': (forecast_time_idx_cum, cum_target_cts),
                        'Predicted (mean)': (forecast_time_idx_cum, cum_pred_cts),
                    }
                    default_x_cum = forecast_time_idx_cum
                    
                    # Add ensemble cumulative returns (using LOG RETURNS)
                    if ensemble_log_ts is not None:
                        for sample_idx in range(n_samples):
                            # Continue from last history cumulative value
                            cum_sample_forecast = last_cum_hist + torch.cumsum(ensemble_log_ts[sample_idx, :, s], dim=0)
                            cum_sample_cts = torch.cat((last_cum_hist.unsqueeze(0), cum_sample_forecast), dim=0)
                            series_cum[f'Sample {sample_idx+1}'] = (forecast_time_idx_cum, cum_sample_cts)

                    marker_cum = {"Observed History": "o", "Target": "d", "Predicted (mean)": "x"}
                    linecolor_cum = {"Observed History": "black", "Target": "tab:blue", "Predicted (mean)": "tab:orange"}
                    linestyle_cum = {"Observed History": "-", "Target": "-", "Predicted (mean)": "--"}
                    
                    if ensemble_ts is not None:
                        for sample_idx in range(n_samples):
                            marker_cum[f'Sample {sample_idx+1}'] = None
                            linecolor_cum[f'Sample {sample_idx+1}'] = "tab:orange"
                            linestyle_cum[f'Sample {sample_idx+1}'] = "-"
                    
                    self._plot_time_series(
                        ax_cum,
                        time_idx,
                        series_cum,
                        ylabel='Cumulative\nLog Returns',
                        title=None,
                        marker=marker_cum,
                        linecolor=linecolor_cum,
                        linestyle=linestyle_cum,
                        legend=(plot_idx == 0),  # Show legend only on first plot
                    )
                    
                    # Add confidence bands for cumulative LOG RETURNS (only for forecast portion)
                    if ensemble_log_ts is not None and show_confidence_bands:
                        # Compute cumulative returns for all ensemble members (LOG RETURNS) - forecast only
                        cum_ensemble_forecast = []
                        for sample_idx in range(n_samples):
                            # Continue from last history cumulative value
                            cum_sample_forecast = last_cum_hist + torch.cumsum(ensemble_log_ts[sample_idx, :, s], dim=0)
                            cum_sample_cts = torch.cat((last_cum_hist.unsqueeze(0), cum_sample_forecast), dim=0)
                            cum_ensemble_forecast.append(cum_sample_cts)
                        
                        cum_ensemble_forecast = torch.stack(cum_ensemble_forecast, dim=0)  # (n_samples, 1+T)
                        
                        # Compute quantiles
                        lower_q = confidence_alpha / 2
                        upper_q = 1 - confidence_alpha / 2
                        cum_lower = torch.quantile(cum_ensemble_forecast, lower_q, dim=0)
                        cum_upper = torch.quantile(cum_ensemble_forecast, upper_q, dim=0)
                        
                        # Time indices for band (forecast portion only)
                        band_x_cum = forecast_time_idx_cum
                        
                        # Plot shaded band
                        ax_cum.fill_between(
                            band_x_cum,
                            cum_lower.cpu().numpy(),
                            cum_upper.cpu().numpy(),
                            color='tab:orange',
                            alpha=0.2,
                            label=f'{int((1-confidence_alpha)*100)}% CI' if plot_idx == 0 else None
                        )
                        
                        # Update legend if we added the band
                        if plot_idx == 0:
                            ax_cum.legend(loc='upper left')
                else:
                    # Skip cumulative plot if we don't have historical log returns
                    # (We need hist_log_ts to properly compute cumulative returns)
                    pass

        axes[-1].set_xlabel('Trading Days')
        # plt.tight_layout()

        # Save figure if requested or if Hydra is active
        if save_dir is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_cfg = HydraConfig.get()
                save_dir = hydra_cfg.runtime.output_dir
            except Exception:
                save_dir = None
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Save with batch index and stock timestamp if available
            if time_start is not None:
                save_path = os.path.join(save_dir, f"stock_forecast_examples_t{time_start}_b{batch_index}.pdf")
            else:
                save_path = os.path.join(save_dir, f"stock_forecast_examples_b{batch_index}.pdf")
                
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved prediction visualization to: {save_path}")

        else:
            logger.info(f"No save_dir provided and Hydra not active; skipping figure save.")

        return fig
    
    # @jaxtyped(typechecker=beartype)
    def _compute_stock_specific_metrics(
        self,
        # pred: Float[torch.Tensor, "batch timesteps nodes features"],
        # target: Float[torch.Tensor, "batch timesteps nodes features"],
        pred: torch.FloatTensor,
        target: torch.FloatTensor,
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        
        """Compute stock-specific evaluation metrics."""
        metrics = {}
        
        # 1. Direction Accuracy (most important for trading)
        prev_prices_y = metadata.get("prev_prices_y")
        if prev_prices_y is not None:

            # Ensure prev_prices has correct shape
            # Simpler (since prepare_data always ensures [B, N, 1] or [B, 1, N, 1])
            if prev_prices_y.dim() == 3:  # [B, N, 1]
                prev_prices_y = prev_prices_y.unsqueeze(1)  # [B, 1, N, 1]
            elif prev_prices_y.dim() == 2:  # [B*N, 1] (fallback)
                B = pred.size(0)
                N = pred.size(2) if pred.dim() == 4 else 1
                prev_prices_y = prev_prices_y.view(B, N, -1).unsqueeze(1)

            # if prev_prices_y.dim() == 3: # [B, N, 1]
            #     if pred.dim() == 4:  # [B, T, N, F]
            #         prev_prices_y = prev_prices_y.unsqueeze(1)  # [B, 1, N, 1]
            #     else:  # [B*N, T, F]
            #         prev_prices_y = prev_prices_y.view(-1, prev_prices_y.size(-1)).unsqueeze(1)  # [B*N, 1, 1]
                
            # elif prev_prices_y.dim() == 2: # [B*N, 1]
            #     if pred.dim() == 4:  # [B, T, N, F]
            #         B_N, N = prev_prices_y.size(0), pred.size(2)
            #         B = B_N // (pred.size(2))
            #         prev_prices_y = prev_prices_y.view(B, N, -1).unsqueeze(1)  # [B, 1, N, 1]
            #     else:  # [B*N, T, F]
            #         prev_prices_y = prev_prices_y.unsqueeze(1)  # [B*N, 1, 1]

            # else:
            #     prev_prices_y = prev_prices_y  # Assume already correct shape
            

            logger.info(f"Prev prices shape for direction accuracy: {prev_prices_y.shape}")
            logger.info(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
            
            # Match shapes for broadcasting
            if pred.dim() == 4:  # [B, T, N, F]
                # Compare each timestep with previous close price
                pred_direction = (pred[:, :, :, :1] > prev_prices_y).float()  # Use first feature (close price)
                target_direction = (target[:, :, :, :1] > prev_prices_y).float()
            else:  # [B*N, T, F]
                B_N, T, F = pred.shape
                prev_prices_flat = prev_prices_y.reshape(-1, 1, 1)  # [B, 1, N, 1] -> [B*N, 1, 1]
                pred_direction = (pred[:, :, :1] > prev_prices_flat).float()
                target_direction = (target[:, :, :1] > prev_prices_flat).float()
            
            direction_accuracy = (pred_direction == target_direction).float().mean().item()
            metrics["direction_accuracy"] = direction_accuracy
        
        # 2. Return prediction error
        if prev_prices_y is not None:
            # Predicted returns
            pred_returns = (pred - prev_prices_y) / (prev_prices_y.abs() + 1e-8)
            target_returns = (target - prev_prices_y) / (prev_prices_y.abs() + 1e-8)
            
            metrics["return_mse"] = torch.nn.functional.mse_loss(pred_returns, target_returns).item()
            metrics["return_mae"] = torch.nn.functional.l1_loss(pred_returns, target_returns).item()
        
        # 3. Percentage error
        percentage_error = torch.abs((target - pred) / (target.abs() + 1e-8)) * 100
        metrics["mape"] = percentage_error.mean().item()
        metrics["median_ape"] = percentage_error.median().item()
        
        # 4. Max error (worst case)
        metrics["max_error"] = torch.abs(target - pred).max().item()
        
        # 5. Relative error (normalized by target magnitude)
        relative_error = torch.abs(target - pred) / (torch.abs(target) + 1e-8)
        metrics["relative_error"] = relative_error.mean().item()
        
        return metrics
    
    def _compute_temporal_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Compute error at different forecast horizons.
        
        Args:
            pred: [B, T, N, F]
            target: [B, T, N, F]
            prefix: Optional prefix for metric names
        """
        metrics = {}
        T = pred.size(1)
        
        for t in range(T):
            # Error at timestep t
            pred_t = pred[:, t]
            target_t = target[:, t]
            
            mse_t = torch.nn.functional.mse_loss(pred_t, target_t).item()
            mae_t = torch.nn.functional.l1_loss(pred_t, target_t).item()
            
            metrics[f"{prefix}mse_horizon_{t+1}"] = mse_t
            metrics[f"{prefix}mae_horizon_{t+1}"] = mae_t
        
        # Average error over short/medium/long term
        if T >= 3:
            short_term = slice(0, T//3)
            medium_term = slice(T//3, 2*T//3)
            long_term = slice(2*T//3, T)
            
            metrics[f"{prefix}mse_short_term"] = torch.nn.functional.mse_loss(
                pred[:, short_term], target[:, short_term]
            ).item()
            metrics[f"{prefix}mse_medium_term"] = torch.nn.functional.mse_loss(
                pred[:, medium_term], target[:, medium_term]
            ).item()
            metrics[f"{prefix}mse_long_term"] = torch.nn.functional.mse_loss(
                pred[:, long_term], target[:, long_term]
            ).item()
        
        return metrics
    
    # ==================================================================
    # Structural metric helpers (shared by V1 and V2 evaluators)
    # ==================================================================

    @staticmethod
    def _lag1_acf(returns: torch.Tensor, squared: bool = True) -> float:
        """Lag-1 normalized autocorrelation, per-stock then averaged.

        When squared=True: ACF of r² (volatility clustering).
        When squared=False: ACF of raw r (momentum / mean-reversion).

        Args:
            returns: [B, T, N, F] or [B, n, T, N, F] or [B_flat, T, F]

        Returns:
            Scalar: mean lag-1 ACF across stocks (and ensemble members).
        """
        if returns.dim() == 5:
            B, n, T, N, F = returns.shape
            x = returns.reshape(B * n, T, N * F)
        elif returns.dim() == 4:
            B, T, N, F = returns.shape
            x = returns.reshape(B, T, N * F)
        elif returns.dim() == 3:
            x = returns
        else:
            raise ValueError(f"_lag1_acf expects 3D/4D/5D input, got {returns.dim()}D")

        if x.shape[1] < 2:
            return 0.0

        if squared:
            x = x ** 2

        mu = x.mean(dim=(0, 1), keepdim=True)              # [1, 1, S]
        c = x - mu                                           # [M, T, S]

        gamma_0 = (c ** 2).mean(dim=(0, 1))                  # [S]
        gamma_1 = (c[:, :-1] * c[:, 1:]).mean(dim=(0, 1))    # [S]

        valid = gamma_0 > 1e-8
        if not valid.any():
            return 0.0
        rho_1 = gamma_1[valid] / gamma_0[valid]
        return rho_1.mean().item()

    @staticmethod
    def _excess_kurtosis(returns: torch.Tensor) -> float:
        """Excess kurtosis of returns, per-stock (F=0) then averaged.

        Real daily returns typically have excess kurtosis ~5-20.
        Gaussian returns have excess kurtosis = 0.

        Args:
            returns: [B, T, N, F] or [B, n, T, N, F] or [B_flat, T, F]
        """
        if returns.dim() == 5:
            B, n, T, N, F = returns.shape
            x_flat = returns[:, :, :, :, 0].reshape(-1, N)
        elif returns.dim() == 4:
            B, T, N, F = returns.shape
            x_flat = returns[:, :, :, 0].reshape(-1, N)
        elif returns.dim() == 3:
            # [B_flat, T, F] — use F=0 only, consistent with 4D/5D
            x_flat = returns[:, :, 0].reshape(-1, 1)
        else:
            raise ValueError(f"_excess_kurtosis expects 3D/4D/5D input, got {returns.dim()}D")

        mu = x_flat.mean(dim=0)
        c = x_flat - mu
        m2 = (c ** 2).mean(dim=0)
        m4 = (c ** 4).mean(dim=0)

        valid = m2 > 1e-8
        if not valid.any():
            return 0.0
        kurt = m4[valid] / (m2[valid] ** 2) - 3.0
        return kurt.mean().item()

    @staticmethod
    def _top1_eigenvalue(returns: torch.Tensor) -> Optional[float]:
        """Top eigenvalue of the return correlation matrix (F=0).

        Only accepts 4D [B, T, N, F] or 5D [B, n, T, N, F] inputs.
        Returns None for degenerate inputs (3D, N < 2, or samples < 2).
        """
        if returns.dim() == 5:
            B, n, T, N, F = returns.shape
            x = returns[:, :, :, :, 0].reshape(-1, N)
        elif returns.dim() == 4:
            B, T, N, F = returns.shape
            x = returns[:, :, :, 0].reshape(-1, N)
        else:
            return None

        if N < 2 or x.shape[0] < 2:
            return None

        x = x.float()
        if not torch.isfinite(x).all():
            return None

        x = x - x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        x = x / (std + 1e-8)
        C = (x.T @ x) / x.shape[0]
        vals = torch.linalg.eigvalsh(C)
        top = vals[-1].item()
        return top if np.isfinite(top) else None

    @classmethod
    def compute_stylized_facts_from_tensors(
        cls,
        cat_real: torch.Tensor,
        cat_gen: torch.Tensor,
    ) -> Dict[str, float]:
        """Recompute kurtosis / vol-clustering / momentum / top-eigval on
        full concatenated tensors.

        These statistics are non-linear (4th moment, ratios) so a per-batch
        average is *not* equivalent to computing them once over the full
        dataset.  Baselines that accumulate raw return tensors across
        evaluation batches should call this helper after the loop and
        overwrite the per-batch averaged values in their metrics dict.

        Sample-size handling differs across stats:

        * ``kurtosis_*``, ``volclustering_*``, ``momentum_*`` — per-stock
          statistics averaged across stocks.  Variance shrinks with more
          samples; bias is independent of sample size.  Pooling all
          generated trajectories (``cat_gen``) gives a tight, unbiased
          estimate of the model's distribution.

        * ``eigval1_*`` — top eigenvalue of the empirical cross-stock
          correlation matrix.  Strongly biased upward when ``# trajectories
          < N_stocks`` (rank-deficient regime).  ``cat_real`` is stuck in
          this regime (one trajectory per window).  Comparing it against
          a pooled ``cat_gen`` (many more samples, lower bias) is
          apples-to-oranges.  Instead, ``cat_gen`` is partitioned into
          ``n_samples_per_window`` non-overlapping sample-size-matched
          subsamples (each containing one sample per window, using a
          different ensemble member), ``eigval1`` is computed on each, and
          the values are averaged.  Each subsample has the same
          rank-deficient bias as ``cat_real`` (fair comparison) and the
          mean has ~``1/sqrt(n_samples_per_window)`` times the variance of
          any single subsample.

        Layout assumption: ``cat_gen`` is expected to be laid out with
        ensemble members interleaved per window — i.e. element ``k``
        corresponds to ``(window=k // n_samples_per_window, sample=k %
        n_samples_per_window)``.  This matches the convention used by
        ``ReplicatedDataset`` and the GRW / Diffusion baselines.  Under
        this layout, ``cat_gen[i::n_samples_per_window]`` selects ensemble
        member ``i`` from every window — exactly the sample-size-matched
        subsample we need.

        Args:
            cat_real: ``[B_real, T, N]`` real returns concatenated across
                evaluation batches.  Each window contributes one trajectory.
            cat_gen: ``[B_real * n_samples, T, N]`` generated returns
                pooled across the full ensemble (interleaved layout).
                When ``B_pool == B_real`` (no replication), the eigval1
                bootstrap reduces to a single chunk = the whole tensor.

        Returns:
            Dict with keys ``kurtosis_gen``, ``kurtosis_real``,
            ``kurtosis_gap``, ``volclustering_gen``, ``volclustering_real``,
            ``volclustering_gap``, ``momentum_gen``, ``momentum_real``,
            ``momentum_gap``, and (when computable) ``eigval1_gen``,
            ``eigval1_real``, ``eigval1_ratio``.
        """
        # Helpers expect 4D [B, T, N, F]; lift the last dim to F=1.
        if cat_real.dim() == 3:
            cat_real = cat_real.unsqueeze(-1)
        if cat_gen.dim() == 3:
            cat_gen = cat_gen.unsqueeze(-1)

        out: Dict[str, float] = {}

        # Per-stock-then-averaged stats — pool full ensemble for low variance.
        out["volclustering_gen"]  = cls._lag1_acf(cat_gen,  squared=True)
        out["volclustering_real"] = cls._lag1_acf(cat_real, squared=True)
        out["volclustering_gap"]  = abs(out["volclustering_gen"] - out["volclustering_real"])

        out["momentum_gen"]  = cls._lag1_acf(cat_gen,  squared=False)
        out["momentum_real"] = cls._lag1_acf(cat_real, squared=False)
        out["momentum_gap"]  = abs(out["momentum_gen"] - out["momentum_real"])

        out["kurtosis_gen"]  = cls._excess_kurtosis(cat_gen)
        out["kurtosis_real"] = cls._excess_kurtosis(cat_real)
        out["kurtosis_gap"]  = abs(out["kurtosis_gen"] - out["kurtosis_real"])

        # Top eigenvalue: bootstrap sample-size-matched subsamples of cat_gen
        # so each subsample has the same rank-deficient bias as cat_real.
        ev_real = cls._top1_eigenvalue(cat_real)
        if ev_real is not None:
            out["eigval1_real"] = ev_real

        B_real = int(cat_real.shape[0])
        B_pool = int(cat_gen.shape[0])
        n_per_window = max(1, B_pool // max(B_real, 1))

        ev_gens: List[float] = []
        for i in range(n_per_window):
            chunk = cat_gen[i::n_per_window][:B_real]
            if int(chunk.shape[0]) < B_real:
                continue  # tail chunk too small — skip rather than bias the mean
            ev = cls._top1_eigenvalue(chunk)
            if ev is not None:
                ev_gens.append(float(ev))
        if ev_gens:
            ev_gen_mean = sum(ev_gens) / len(ev_gens)
            out["eigval1_gen"] = ev_gen_mean
            out["eigval1_gen_n_chunks"] = float(len(ev_gens))
            if len(ev_gens) > 1:
                # unbiased SE of the mean across the n_chunks subsamples
                m = ev_gen_mean
                var = sum((v - m) ** 2 for v in ev_gens) / (len(ev_gens) - 1)
                out["eigval1_gen_se"] = (var / len(ev_gens)) ** 0.5
            if ev_real is not None and ev_real >= 0.01:
                out["eigval1_ratio"] = ev_gen_mean / ev_real

        return out

    def _compute_return_metrics(
        self,
        pred_returns: torch.Tensor,
        target_returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute secondary diagnostic metrics on log returns.

        Args:
            pred_returns: [B, T, N, F] or [B*N, T, F]
            target_returns: [B, T, N, F] or [B*N, T, F]
        """
        metrics = {}
        
        # Basic return prediction metrics
        metrics["return_mse"] = torch.nn.functional.mse_loss(pred_returns, target_returns).item()
        metrics["return_mae"] = torch.nn.functional.l1_loss(pred_returns, target_returns).item()
        metrics["return_rmse"] = torch.sqrt(torch.nn.functional.mse_loss(pred_returns, target_returns)).item()
        
        # Direction accuracy on returns (up/down prediction)
        pred_direction = (pred_returns > 0).float()
        target_direction = (target_returns > 0).float()
        metrics["return_direction_accuracy"] = (pred_direction == target_direction).float().mean().item()
        
        # Structural metrics on generated vs target returns
        metrics["volclustering_gen"] = self._lag1_acf(pred_returns, squared=True)
        metrics["volclustering_real"] = self._lag1_acf(target_returns, squared=True)
        metrics["volclustering_gap"] = abs(metrics["volclustering_gen"] - metrics["volclustering_real"])

        metrics["momentum_gen"] = self._lag1_acf(pred_returns, squared=False)
        metrics["momentum_real"] = self._lag1_acf(target_returns, squared=False)
        metrics["momentum_gap"] = abs(metrics["momentum_gen"] - metrics["momentum_real"])

        metrics["kurtosis_gen"] = self._excess_kurtosis(pred_returns)
        metrics["kurtosis_real"] = self._excess_kurtosis(target_returns)
        metrics["kurtosis_gap"] = abs(metrics["kurtosis_gen"] - metrics["kurtosis_real"])

        ev_gen = self._top1_eigenvalue(pred_returns)
        ev_real = self._top1_eigenvalue(target_returns)
        if ev_gen is not None:
            metrics["eigval1_gen"] = ev_gen
        if ev_real is not None:
            metrics["eigval1_real"] = ev_real
        if ev_gen is not None and ev_real is not None and ev_real >= 0.01:
            metrics["eigval1_ratio"] = ev_gen / ev_real

        return metrics


def _compute_direction_accuracy_pair(
    target: torch.Tensor,
    gen_ensemble: Optional[torch.Tensor],
    pred_mean: torch.Tensor,
    threshold,
    ensemble_dim: int = 1,
) -> Tuple[float, Optional[float]]:
    """Compute (per-sample, majority-vote) direction-accuracy given a threshold.

    ``target`` is the ground-truth tensor whose direction is being predicted;
    ``gen_ensemble`` (if not None and has >1 sample along ``ensemble_dim``) is
    used for both the per-sample and majority-vote computations; otherwise
    ``pred_mean`` is used for the single-sample fallback.

    A point is "up" iff its value > ``threshold``. ``threshold`` can be a
    Python scalar (e.g., ``0.0``) or a tensor that broadcasts with both
    ``target`` and ``gen_ensemble`` / ``pred_mean``.

    Returns:
        per_sample_acc: float in [0, 1]
        majority_acc:   float in [0, 1] if an ensemble was used, else None
    """
    target_anomaly = (target > threshold).float()

    if gen_ensemble is not None and gen_ensemble.shape[ensemble_dim] > 1:
        ens_anomaly = (gen_ensemble > threshold).float()
        target_anomaly_expanded = target_anomaly.unsqueeze(ensemble_dim)
        per_sample_acc = (ens_anomaly == target_anomaly_expanded).float().mean().item()
        vote_frac = ens_anomaly.mean(dim=ensemble_dim)
        majority_pred = (vote_frac > 0.5).float()
        majority_acc = (majority_pred == target_anomaly).float().mean().item()
        return per_sample_acc, majority_acc

    pred_anomaly = (pred_mean > threshold).float()
    per_sample_acc = (pred_anomaly == target_anomaly).float().mean().item()
    return per_sample_acc, None


@TASK_REGISTRY.register("stock_price_forecasting_v2")
class StockPriceForecastingTaskV2(StockPriceForecastingTask):
    """
    Version 2 of Stock Price Forecasting Task evaluator.
    
    Designed for SP500 dataset experiments while maintaining backward compatibility with SP100.
    Adds SP500-specific de-standardization pipeline to handle per-stock standardized features.
    
    Key Features:
        - Handles SP500 per-stock standardization metadata (mean/std for each feature)
        - De-standardizes model outputs before price conversion
        - Backward compatible with SP100 (uses scale factors if no per_stock_stats)
    
    Usage:
        In your config, set: task.name: stock_price_forecasting_v2
    """
    
    def __init__(
        self,
        forecast_horizon: int = 5,
        normalizer: Optional[Normalizer] = None,
        n_samples_per_input: int = 20,
        **kwargs
    ):
        # Initialize with SP500 as default valid dataset
        if 'valid_datasets' not in kwargs:
            kwargs['valid_datasets'] = ['sp500', 'sp100']  # SP500 first for priority
        
        super().__init__(
            forecast_horizon=forecast_horizon,
            normalizer=normalizer,
            n_samples_per_input=n_samples_per_input,
            **kwargs
        )
        
        # Storage for SP500 standardization metadata (injected by trainer)
        self.dataset_info: Optional[Dict] = None

        # NEW: Target destandardization support (injected via set_target_destandardization)
        self.target_stats = None
        self.destandardize_target_fn = None

        # Post-hoc variance correction config, transiently injected by
        # DiffusionBaseline.evaluate() (see
        # docs/agent_summaries/variance_correction_plan_2026-05-20.md).
        # None or `mode=='off'` → apply_variance_correction is a no-op.
        # See `apply_variance_correction` for the expected attribute contract
        # (`.mode`, `.alpha`, `.alpha_per_horizon`).
        self.variance_correction: Optional[Any] = None

        logger.info(f"✅ Initialized StockPriceForecastingTaskV2 for datasets: {self.valid_datasets}")
    
    def set_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """
        Inject dataset metadata for SP500 per-stock standardization.
        
        Called by CLI after task instantiation to provide standardization
        metadata computed by the dataset builder.
        
        Args:
            dataset_info: Dict containing required and optional fields:
            
            **Required fields:**
                - per_stock_stats: Dict[feature_name -> {'mean': np.ndarray[N], 'std': np.ndarray[N]}]
                    Per-stock standardization statistics for each feature.
                - feature_names: List[str]
                    Ordered list of all feature names matching dataset columns.
                - target_feature_idx: int
                    Index of target feature in feature_names list (e.g., 0 for 'DailyLogReturn').
            
            **Optional fields:**
                - stock_symbols: List[str]
                    Ordered list of stock symbols matching stat arrays (for logging).
                - standardized_features: List[str]
                    Subset of features that were standardized.
                - sp500_scale_factor: float
                    Legacy scale factor if needed.
        
        Raises:
            ValueError: If required fields are missing during evaluation.
        """
        self.dataset_info = dataset_info
        logger.info(f"✅ Dataset info injected into StockPriceForecastingTaskV2")
        
        if 'per_stock_stats' in dataset_info:
            logger.info(f"   Found per_stock_stats for {len(dataset_info['per_stock_stats'])} features")
            logger.info(f"   Target feature index: {dataset_info.get('target_feature_idx', 'NOT SET')}")
            logger.info(f"   Feature names: {dataset_info.get('feature_names', [])}")
    
    def set_target_destandardization(
        self, 
        target_stats: Dict[str, Any],
        destandardize_fn: callable
    ) -> None:
        """
        Inject target destandardization method and stats (new SP500 standardization API).
        
        This provides a simpler alternative to the legacy per_stock_stats pipeline.
        
        Args:
            target_stats: Dict with keys:
                - 'mean': torch.Tensor[N] per-stock means
                - 'std': torch.Tensor[N] per-stock stds
                - 'scale_factor': float (e.g., 100.0 for return features)
                - 'target_name': str (e.g., 'DailyLogReturn')
            destandardize_fn: Static method for destandardization 
                             (e.g., SP500Stocks.destandardize_target)
        """
        self.target_stats = target_stats
        self.destandardize_target_fn = destandardize_fn
        
        logger.info(f"✅ Target destandardization injected into {self.__class__.__name__}")
        logger.info(f"   Target: {target_stats['target_name']}")
        logger.info(f"   Scale factor: {target_stats['scale_factor']}")
        logger.info(f"   Per-stock stats: mean shape={target_stats['mean'].shape}, std shape={target_stats['std'].shape}")
    
    def _destandardize_samples(
        self,
        standardized_samples: torch.Tensor,
        per_stock_stats: Dict[str, Dict[str, np.ndarray]],
        stock_symbols: List[str],
        feature_idx: int,
        stock_dim: int = -2
    ) -> torch.Tensor:
        """
        De-standardize samples using per-stock statistics.
        
        Reverses z-scoring: x = z * std + mean (per stock, per feature)
        
        Args:
            standardized_samples: Standardized log returns, any shape with stock dimension
                                 Examples: [B, T, N, F], [B*N, T, F], [B, n, T, N, F]
            per_stock_stats: Dict mapping feature names to {'mean': [N,], 'std': [N,]} arrays
            stock_symbols: List of stock ticker symbols (len N)
            feature_idx: Which feature to de-standardize (usually 0 for DailyLogReturn)
            stock_dim: Dimension index for stocks (default -2 for [..., N, F])
        
        Returns:
            De-standardized samples (raw log returns with std ~0.02)
        
        Example:
            # For SP500 DailyLogReturn (feature_idx=0):
            raw_log_returns = _destandardize_samples(
                standardized_samples=model_outputs,  # [B, T, N, F] with std ~1.0
                per_stock_stats=self.dataset_info['per_stock_stats'],
                stock_symbols=self.dataset_info['stock_symbols'],
                feature_idx=0
            )
            # Output: raw_log_returns with std ~0.02
        """
        # Get feature name from feature_idx
        feature_names = self.dataset_info.get('feature_names', 
                                              ['DailyLogReturn', 'ALR1', 'ALR5', 'ALR21', 'MACD'])
        if feature_idx >= len(feature_names):
            raise ValueError(f"feature_idx {feature_idx} out of range for {len(feature_names)} features")
        
        feature_name = feature_names[feature_idx]
        
        # Check if this feature needs de-standardization
        standardized_features = self.dataset_info.get('standardized_features', [])
        if feature_name not in standardized_features:
            logger.info(f"Feature '{feature_name}' not in standardized_features list - skipping de-standardization")
            return standardized_samples
        
        # Get statistics for this feature
        if feature_name not in per_stock_stats:
            raise KeyError(f"Feature '{feature_name}' not found in per_stock_stats. "
                          f"Available: {list(per_stock_stats.keys())}")
        
        stats = per_stock_stats[feature_name]
        means = torch.from_numpy(stats['mean']).to(standardized_samples.device)  # [N,]
        stds = torch.from_numpy(stats['std']).to(standardized_samples.device)    # [N,]
        
        # Verify stock count matches
        N_stats = len(means)
        N_samples = standardized_samples.shape[stock_dim]
        if N_stats != N_samples:
            raise ValueError(f"Stock count mismatch: stats has {N_stats}, samples has {N_samples}")
        
        # Reshape stats to match samples shape
        # For stock_dim=-2 with shape [B, T, N, F]: reshape to [1, 1, N, 1]
        # For stock_dim=0 with shape [N, T, F]: reshape to [N, 1, 1]
        target_shape = [1] * standardized_samples.ndim
        target_shape[stock_dim] = N_stats
        
        means_expanded = means.view(*target_shape)  # [..., N, ...]
        stds_expanded = stds.view(*target_shape)    # [..., N, ...]
        
        # De-standardize: x = z * std + mean
        destandardized = standardized_samples * stds_expanded + means_expanded
        
        logger.info(f"De-standardized '{feature_name}': "
                   f"std {standardized_samples.std().item():.4f} → {destandardized.std().item():.4f}")
        
        return destandardized
    
    def prepare_data(
        self,
        data_batch: Union[StocksDataDiffusion, Batch],
        skip_descaling: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare data for evaluation with SP500 de-standardization support.
        
        Overrides parent to add SP500-specific de-standardization:
        1. Extract standardized targets from batch
        2. De-standardize using per_stock_stats if available (SP500)
        3. Fall back to scale_factors if no stats (SP100)
        4. Convert to prices
        
        Args:
            data_batch: Batch of graph data with targets
            skip_descaling: If True, skip both de-standardization and scale factor de-scaling
        
        Returns:
            Dict with 'samples' (de-standardized) and 'metadata'
        """
        # Call parent implementation first
        result = super().prepare_data(data_batch, **kwargs)
        
        samples = result['samples']  # [B, T, N, F] - currently standardized if SP500
        metadata = result['metadata']
        
        # Check if we have SP500 standardization metadata (injected by trainer)
        if self.dataset_info is not None and 'per_stock_stats' in self.dataset_info:
            logger.info("🔍 Detected SP500 per_stock_stats - applying de-standardization...")
            
            per_stock_stats = self.dataset_info['per_stock_stats']
            stock_symbols = self.dataset_info.get('stock_symbols', [])
            feature_idx = 0  # DailyLogReturn is first feature
            
            # De-standardize targets (ground truth)
            samples = self._destandardize_samples(
                standardized_samples=samples,
                per_stock_stats=per_stock_stats,
                stock_symbols=stock_symbols,
                feature_idx=feature_idx,
                stock_dim=-2  # [B, T, N, F]
            )
            
            # Update metadata to indicate de-standardization was applied
            metadata['destandardized_sp500'] = True
            metadata['raw_log_return_std'] = samples.std().item()
            logger.info(f"   Target samples de-standardized: std={samples.std().item():.6f} (expected ~0.02)")
            
            # Also de-standardize historical log returns if present
            if 'hist_log_returns' in metadata and metadata['hist_log_returns'] is not None:
                hist_log = metadata['hist_log_returns']  # [B, T_hist, N, F]
                metadata['hist_log_returns'] = self._destandardize_samples(
                    standardized_samples=hist_log,
                    per_stock_stats=per_stock_stats,
                    stock_symbols=stock_symbols,
                    feature_idx=feature_idx,
                    stock_dim=-2
                )
                logger.info(f"   Historical samples de-standardized: std={metadata['hist_log_returns'].std().item():.6f}")
        
        else:
            logger.info("No SP500 per_stock_stats found - using default scale factor de-scaling (SP100 mode)")
            metadata['destandardized_sp500'] = False
        
        # Update result with de-standardized samples
        result['samples'] = samples
        result['metadata'] = metadata
        
        return result
    
    def prepare_data_v2(self, data_batch: Union[StocksDataDiffusion, Batch]) -> Dict[str, Any]:
        """
        Extract raw standardized samples and metadata (V2 - minimal transformations).
        
        Returns samples in standardized space without any destandardization.
        Destandardization happens in evaluate_samples().
        
        Args:
            data_batch: PyG Data with attributes:
                - y: [B*n, T, F] target samples (STANDARDIZED)
                - x: [B*n, T, F] historical samples (STANDARDIZED)
                - close_price: [B*n, T, 1] historical closing prices
                - close_price_y: [B*n, T, 1] target closing prices
                - edge_index: [2, E]
                - edge_weight: [E]
                - stocks_index: [(B*n)*N] stock identifiers
                - timestamp: [B*n]
        
        Returns:
            Dict with:
                - 'samples': Standardized target samples [B*n, T, N, F]
                - 'metadata': Dict with edge_index, stocks_index, close_price, etc.,
                             plus shape info (batch_size, num_timesteps, num_stocks, num_features)
        """
        logger.info(f"📊 Using prepare_data_v2 (minimal transformations)")
        
        # Extract raw standardized samples
        # CRITICAL: SP500 dataset returns per-stock samples where each row is a stock
        # Shape is [B*n*N, T, F] NOT [B*n, T, F]
        samples = data_batch.y  # [B*n*N, T, F]
        hist_samples = data_batch.x if hasattr(data_batch, 'x') else None  # [B*n*N, T_hist, F] historical observations
        stocks_index = data_batch.stocks_index if hasattr(data_batch, 'stocks_index') else None
        
        # Reshape from per-stock format [B*n*N, T, F] to batched format [B*n, T, N, F]
        # IMPORTANT: Use [B*n, T, N, F] format to match parent class pipeline expectations
        if samples.dim() == 3 and stocks_index is not None:
            total_entries = samples.shape[0]  # B*n*N
            num_timesteps = samples.shape[1]  # T
            num_features = samples.shape[2]  # F
            
            # Find num_stocks by checking unique values in stocks_index
            # stocks_index pattern: [0,1,...,N-1, 0,1,...,N-1, ...] repeated B*n times
            num_stocks = len(stocks_index.unique())
            batch_size = total_entries // num_stocks  # B*n
            
            # Reshape: [B*n*N, T, F] -> [B*n, N, T, F] -> [B*n, T, N, F]
            samples = samples.view(batch_size, num_stocks, num_timesteps, num_features)
            samples = samples.transpose(1, 2)  # [B*n, N, T, F] -> [B*n, T, N, F]
            logger.info(f"   Reshaped samples: [B*n*N={total_entries}, T={num_timesteps}, F={num_features}] -> "
                       f"[B*n={batch_size}, T={num_timesteps}, N={num_stocks}, F={num_features}]")
        
        # Extract shape info for metadata
        batch_size = samples.shape[0]  # B*n
        num_timesteps = samples.shape[1]  # T
        num_stocks = samples.shape[2]  # N
        num_features = samples.shape[3]  # F
        
        # Defensively validate shapes
        assert samples.shape == (batch_size, num_timesteps, num_stocks, num_features), \
            f"Shape mismatch: got {samples.shape}, expected [{batch_size}, {num_timesteps}, {num_stocks}, {num_features}]"
        
        if stocks_index is not None:
            stocks_index_numel = stocks_index.numel()
            expected_numel = batch_size * num_stocks
            assert stocks_index_numel == expected_numel, \
                f"Shape mismatch: stocks_index.numel()={stocks_index_numel} != batch_size*num_stocks={expected_numel}"
        
        logger.info(f"   ✅ Shape validation passed: batch_size={batch_size}, T={num_timesteps}, N={num_stocks}, F={num_features}")
        
        # Reshape historical samples if available
        hist_samples_reshaped = None
        if hist_samples is not None:
            # hist_samples: [B*n*N, T_hist, F] -> [B*n, T_hist, N, F]
            hist_samples_reshaped = hist_samples.view(batch_size, num_stocks, hist_samples.shape[1], hist_samples.shape[2])
            hist_samples_reshaped = hist_samples_reshaped.transpose(1, 2)  # [B*n, T_hist, N, F]
            logger.info(f"   Historical samples reshaped: {hist_samples.shape} -> {hist_samples_reshaped.shape}")
        
        # Build metadata dict
        metadata = {
            # 'edge_index': data_batch.edge_index if hasattr(data_batch, 'edge_index') else None,
            # 'edge_weight': data_batch.edge_weight if hasattr(data_batch, 'edge_weight') else None,
            'stocks_index': data_batch.stocks_index if hasattr(data_batch, 'stocks_index') else None,
            'timestamp': data_batch.timestamp if hasattr(data_batch, 'timestamp') else None,
            'close_price': data_batch.close_price if hasattr(data_batch, 'close_price') else None,
            'close_price_y': data_batch.close_price_y if hasattr(data_batch, 'close_price_y') else None,
            'hist_samples': hist_samples_reshaped,  # [B*n, T_hist, N, F] historical log returns
            # Shape info for downstream processing
            'batch_size': batch_size,
            'num_timesteps': num_timesteps,
            'num_stocks': num_stocks,
            'num_features': num_features,
            # Ensemble info (from task config)
            'n_samples_per_input': self.n_samples_per_input if hasattr(self, 'n_samples_per_input') else None,
        }
        
        logger.info(f"   Metadata keys: {list(metadata.keys())}")
        
        return {
            'samples': samples,  # [B*n, T, N, F] STANDARDIZED (matches parent class format)
            'metadata': metadata
        }
    
    def prepare_data(self, data_batch: Union[StocksDataDiffusion, Batch], **kwargs) -> Dict[str, Any]:
        """Override to route to prepare_data_v2()."""
        return self.prepare_data_v2(data_batch)

    def apply_variance_correction(
        self,
        generated_samples: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        """Post-hoc spread rescaling applied in standardized space.

        Applies ``gen' = gen_mean + alpha * (gen - gen_mean)`` around the
        per-window per-stock per-horizon ensemble mean. The per-ensemble
        mean is preserved exactly; only spread is scaled. See
        ``docs/agent_summaries/variance_correction_plan_2026-05-20.md`` for
        full design rationale (single application point, V2-only,
        before-batch_hook positioning).

        Args:
            generated_samples: ``[B*n, T, N, F]`` standardized model outputs.
            metadata: Dict containing ``n_samples_per_input`` (used to
                derive the ensemble axis n from the leading batch dim).

        Returns:
            ``[B*n, T, N, F]`` tensor with the same shape as the input,
            rescaled around the per-window ensemble mean.

        Raises:
            ValueError: If ``self.variance_correction`` is enabled but
                metadata is missing/invalid, the batch shape contract is
                violated, or the alpha config is malformed.
        """
        vc = getattr(self, "variance_correction", None)
        if vc is None:
            return generated_samples
        mode = str(getattr(vc, "mode", "off") or "off").lower()
        if mode == "off":
            return generated_samples
        pivot = str(getattr(vc, "pivot", "ensemble_mean") or "ensemble_mean").lower()

        Bn, T, N, F = generated_samples.shape
        alpha = self._resolve_alpha_tensor(
            mode=mode,
            vc=vc,
            T=T,
            device=generated_samples.device,
            dtype=generated_samples.dtype,
        )  # [T]

        if pivot == "zero":
            # Global rescale: x' = α(t) · x in standardized space.
            # Equivalent to "pivot around per-stock long-run mean μ_s" in
            # destandardized space (because y = z·σ_s + μ_s ⇒ α·z·σ_s + μ_s
            # = μ_s + α·(y - μ_s)). Structural metrics (γ_k / γ_0 ratios,
            # correlation eigenvalues) are STRICTLY scale-invariant under
            # this transform. Per-window per-horizon ensemble mean moves
            # by (α-1)·μ̂, which is small for daily log returns (model
            # forecasts a few tenths of σ in standardized space). No
            # ensemble axis needed → metadata['n_samples_per_input'] is
            # not consulted in this branch.
            return alpha.view(1, T, 1, 1) * generated_samples

        if pivot != "ensemble_mean":
            raise ValueError(
                "variance_correction.pivot must be 'ensemble_mean' or "
                f"'zero'; got {pivot!r}."
            )

        # Source-of-truth for ensemble grouping. When mode != off and
        # pivot == 'ensemble_mean', require a valid n; do not silently
        # fall back to n=1 (a missing key signals an upstream wiring
        # error and silent no-op would hide it).
        n_raw = metadata.get("n_samples_per_input", None)
        if n_raw is None or int(n_raw) < 1:
            raise ValueError(
                "apply_variance_correction: metadata['n_samples_per_input'] "
                f"is missing or invalid ({n_raw!r}). Variance correction is "
                "enabled with pivot='ensemble_mean' but the ensemble "
                "grouping is undefined. Verify that "
                "task.n_samples_per_input was synced to the baseline's "
                "n_samples before evaluation (see "
                "compare_baselines.py:1705-1708 and the matching sync in "
                "evaluate.py after task = setup_task(...)). Switch to "
                "pivot='zero' to bypass the ensemble axis entirely."
            )
        n = int(n_raw)

        if Bn % n != 0:
            raise ValueError(
                f"apply_variance_correction: Bn={Bn} not divisible by "
                f"n_samples_per_input={n} (from metadata). Batch shape "
                "contract violated — generated_samples must be "
                "[B*n, T, N, F]."
            )

        # NOTE: we do NOT cross-check metadata['batch_size']. In V2's
        # prepare_data_v2 (evaluator.py:2032, :2067) `batch_size` is set
        # to `samples.shape[0]` AFTER the [B*n*N, T, F] → [B*n, T, N, F]
        # reshape — i.e., it is already B*n, not B. The divisibility
        # check above is the only locally-decidable shape contract; an
        # inconsistent n that happens to divide Bn cleanly cannot be
        # caught at this layer and must be prevented upstream by the
        # evaluate.py task sync.
        B = Bn // n

        gen = generated_samples.view(B, n, T, N, F)
        gen_mean = gen.mean(dim=1, keepdim=True)              # [B, 1, T, N, F]
        gen_corrected = gen_mean + alpha.view(1, 1, T, 1, 1) * (gen - gen_mean)
        return gen_corrected.view(Bn, T, N, F)

    def _resolve_alpha_tensor(
        self,
        mode: str,
        vc: Any,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize the per-horizon alpha tensor for the chosen mode.

        ``mode == 'scalar'``: broadcast ``vc.alpha`` (float) across T horizons.
        ``mode == 'per_horizon'``: use ``vc.alpha_per_horizon`` (list[float], len=T).
        """
        if mode == "scalar":
            alpha_scalar = getattr(vc, "alpha", None)
            if alpha_scalar is None:
                raise ValueError(
                    "variance_correction.mode='scalar' but alpha is None. "
                    "Provide a float (e.g., "
                    "++baselines.diffusion.variance_correction.alpha=1.5)."
                )
            return torch.full(
                (T,), float(alpha_scalar), device=device, dtype=dtype,
            )
        if mode == "per_horizon":
            per_h = getattr(vc, "alpha_per_horizon", None)
            if per_h is None:
                raise ValueError(
                    "variance_correction.mode='per_horizon' but "
                    "alpha_per_horizon is None."
                )
            per_h_list = list(per_h)
            if len(per_h_list) != T:
                raise ValueError(
                    "variance_correction.alpha_per_horizon has length "
                    f"{len(per_h_list)} but expected T={T}."
                )
            return torch.tensor(
                [float(x) for x in per_h_list],
                device=device,
                dtype=dtype,
            )
        raise ValueError(
            "variance_correction.mode must be one of 'off', 'scalar', "
            f"'per_horizon'; got {mode!r}."
        )

    def _destandardize_both_samples(
        self,
        generated_samples: torch.Tensor,
        real_samples: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Destandardize both generated and real samples using per-stock statistics.
        
        Args:
            generated_samples: Model outputs [B*n, T, N, F] (STANDARDIZED)
            real_samples: Ground truth [B*n, T, N, F] (STANDARDIZED)
            metadata: Dict with stocks_index, batch_size, num_stocks, n_samples_per_input
        
        Returns:
            Tuple of (generated_destd, real_destd) both with shape [B*n, T, N, F]
        """
        logger.info(f"🔄 Starting destandardization...")
        
        # NEW: Use injected destandardization method if available (simpler API)
        if self.destandardize_target_fn is not None and self.target_stats is not None:
            logger.info(f"   Using injected destandardization method (new API)")
            logger.info(f"   Generated samples shape: {generated_samples.shape}")
            logger.info(f"   Real samples shape: {real_samples.shape}")
            
            # Extract stats and move to correct device
            device = generated_samples.device
            mean = self.target_stats['mean'].to(device)  # [N]
            std = self.target_stats['std'].to(device)    # [N]
            scale_factor = self.target_stats['scale_factor']
            
            logger.info(f"   Stats shapes: mean={mean.shape}, std={std.shape}, scale_factor={scale_factor}")
            
            # Reshape stats for broadcasting with [B*n, T, N, F]
            # mean/std are [N], need to be [1, 1, N, 1] to broadcast with [B*n, T, N, F]
            mean = mean.view(1, 1, -1, 1)  # [1, 1, N, 1]
            std = std.view(1, 1, -1, 1)    # [1, 1, N, 1]
            
            # Destandardize: x = (z * std + mean) * scale_factor
            # where z is standardized (mean=0, std=1), x is raw log return in percentages
            generated_destd = (generated_samples * (std + 1e-8) + mean) * scale_factor
            real_destd = (real_samples * (std + 1e-8) + mean) * scale_factor
            
            logger.info(f"✅ Destandardization complete (new API)")
            logger.info(f"   Generated: mean={generated_destd.mean():.4f}, std={generated_destd.std():.4f}")
            logger.info(f"   Real: mean={real_destd.mean():.4f}, std={real_destd.std():.4f}")
            
            # Store per-stock mean for direction accuracy (de-meaned sign test)
            # mean is [1, 1, N, 1] in destandardised return space (already scaled)
            metadata['per_stock_mean'] = (mean * scale_factor)  # [1, 1, N, 1]
            
            return generated_destd, real_destd
        
        # LEGACY: Fall back to per_stock_stats-based destandardization
        logger.info(f"   Using legacy per_stock_stats destandardization")
        
        # Validate dataset_info
        if self.dataset_info is None:
            raise ValueError(
                "dataset_info is None! TaskV2 requires dataset_info to be injected via set_dataset_info(). "
                "Ensure CLI calls task.set_dataset_info(builder.get_dataset_info()) after task instantiation."
            )
        
        required_fields = ['per_stock_stats', 'feature_names', 'target_feature_idx']
        missing_fields = [f for f in required_fields if f not in self.dataset_info]
        if missing_fields:
            raise ValueError(
                f"dataset_info missing required fields: {missing_fields}. "
                f"Available fields: {list(self.dataset_info.keys())}. "
                f"Required: {required_fields}"
            )
        
        # Extract target feature info
        target_feature_idx = self.dataset_info['target_feature_idx']
        feature_names = self.dataset_info['feature_names']
        per_stock_stats = self.dataset_info['per_stock_stats']
        stock_symbols = self.dataset_info.get('stock_symbols', None)
        
        if target_feature_idx >= len(feature_names):
            raise ValueError(
                f"target_feature_idx={target_feature_idx} out of range for feature_names (len={len(feature_names)})"
            )
        
        target_feature_name = feature_names[target_feature_idx]
        logger.info(f"🎯 Target feature: '{target_feature_name}' (index={target_feature_idx})")
        
        if target_feature_name not in per_stock_stats:
            raise ValueError(
                f"Target feature '{target_feature_name}' not found in per_stock_stats. "
                f"Available features: {list(per_stock_stats.keys())}"
            )
        
        # Extract and validate std/mean arrays
        stats = per_stock_stats[target_feature_name]
        if 'mean' not in stats or 'std' not in stats:
            raise ValueError(
                f"per_stock_stats['{target_feature_name}'] missing 'mean' or 'std'. "
                f"Available keys: {list(stats.keys())}"
            )
        
        # Convert numpy arrays to torch tensors (keep on CPU for indexing)
        mean_array = torch.from_numpy(stats['mean']).float()  # [N,] CPU tensor
        std_array = torch.from_numpy(stats['std']).float()    # [N,] CPU tensor
        
        # Validate std > 0
        zero_std_mask = (std_array == 0)
        if zero_std_mask.any():
            zero_indices = torch.where(zero_std_mask)[0].tolist()
            if stock_symbols is not None and len(stock_symbols) == len(std_array):
                zero_stocks_info = [f"{idx}({stock_symbols[idx]})" for idx in zero_indices]
            else:
                zero_stocks_info = [str(idx) for idx in zero_indices]
            
            raise ValueError(
                f"Zero std detected for feature '{target_feature_name}' at stock indices: {zero_stocks_info}. "
                f"Cannot destandardize with zero std. Check data preprocessing."
            )
        
        logger.info(f"✅ std validation passed: all values > 0")
        logger.info(f"   std range: [{std_array.min().item():.6f}, {std_array.max().item():.6f}]")
        logger.info(f"   mean range: [{mean_array.min().item():.6f}, {mean_array.max().item():.6f}]")
        
        # Validate and reshape stocks_index
        stocks_index = metadata.get('stocks_index')
        if stocks_index is None:
            raise ValueError("metadata missing 'stocks_index' required for destandardization")
        
        batch_size = metadata['batch_size']  # B*n
        num_stocks = metadata['num_stocks']  # N
        
        stocks_index_2d = stocks_index.view(batch_size, num_stocks)  # [B*n, N]
        logger.info(f"📍 stocks_index reshaped: [{stocks_index.numel()},] -> [B*n={batch_size}, N={num_stocks}]")
        
        # Use stocks_index[0, :] since pattern is identical across batches (ReplicatedDataset)
        stock_indices = stocks_index_2d[0, :].long().cpu()  # [N,] - move to CPU for indexing
        
        # Get per-stock statistics for the stocks in this batch
        mean_selected = mean_array[stock_indices]  # [N,]
        std_selected = std_array[stock_indices]    # [N,]
        
        # CRITICAL FIX: Instead of computing statistics, directly compute correct log returns from prices
        close_price = metadata.get('close_price')
        close_price_y = metadata.get('close_price_y')
        
        # VALIDATION: Check if standardization was done correctly using stored per-stock statistics
        if close_price is not None and close_price_y is not None and close_price.dim() == 3 and close_price_y.dim() == 3:
            logger.info(f"🔍 VALIDATING STANDARDIZATION using stored per-stock statistics...")
            
            # Compute actual log returns from prices
            # IMPORTANT: Dataset stores DailyLogReturn as PERCENTAGES (multiplied by 100)
            # See download_sp500_data.py line 135: np.log(...) * 100
            last_hist_price = close_price[:, -1:, :]  # [B_original*N, 1, 1]
            all_prices = torch.cat([last_hist_price, close_price_y], dim=1)  # [B_original*N, T_future+1, 1]
            log_returns_actual = torch.log(all_prices[:, 1:, :] / all_prices[:, :-1, :]) * 100  # [B_original*N, T_future, 1] - AS PERCENTAGES
            
            # Reshape to match real_samples structure: [B*n, T, N, F]
            B_times_n = real_samples.shape[0]
            T = real_samples.shape[1]
            N = real_samples.shape[2]
            
            total_entries = log_returns_actual.shape[0]
            if total_entries == B_times_n * N and log_returns_actual.shape[1] == T:
                # Reshape [B*n*N, T, 1] -> [B*n, N, T, 1] -> [B*n, T, N, 1]
                log_returns_reshaped = log_returns_actual.view(B_times_n, N, T, 1).transpose(1, 2)  # [B*n, T, N, 1]
                
                # Apply stored per-stock standardization: z = (x - mean) / std
                # Broadcast: mean_selected [N,] -> [1, 1, N, 1], log_returns_reshaped [B*n, T, N, 1]
                device = real_samples.device
                mean_broadcasted = mean_selected.to(device).view(1, 1, N, 1)
                std_broadcasted = std_selected.to(device).view(1, 1, N, 1)
                
                expected_standardized = (log_returns_reshaped - mean_broadcasted) / std_broadcasted  # [B*n, T, N, 1]
                
                # Compare with actual real_samples
                diff = (expected_standardized - real_samples).abs().mean().item()
                rel_diff = diff / real_samples.abs().mean().item() * 100 if real_samples.abs().mean().item() > 1e-6 else 0
                
                logger.info(f"   Expected standardized (from applying stored stats to prices):")
                logger.info(f"      mean={expected_standardized.mean().item():.6f}, std={expected_standardized.std().item():.6f}")
                logger.info(f"   Actual real_samples (from dataset):")
                logger.info(f"      mean={real_samples.mean().item():.6f}, std={real_samples.std().item():.6f}")
                logger.info(f"   Difference: abs={diff:.6f}, rel={rel_diff:.2f}%")
                
                if rel_diff < 1:
                    logger.info(f"   ✅ STANDARDIZATION IS CORRECT! Stored per-stock statistics match (rel diff < 1%)")
                    logger.info(f"      Destandardization using stored stats should work perfectly.")
                elif rel_diff < 10:
                    logger.warning(f"   ⚠️  Small discrepancy in standardization (1% < rel diff < 10%)")
                    logger.warning(f"      May be due to numerical precision or slight preprocessing differences.")
                else:
                    logger.error(f"   ❌ STANDARDIZATION IS WRONG! Stored statistics DO NOT match actual data (rel diff > 10%)")
                    logger.error(f"      Data was NOT standardized using the per-stock mean/std in dataset_info!")
                    logger.error(f"      Possible causes:")
                    logger.error(f"         1. Global standardization was used (all stocks same mean/std)")
                    logger.error(f"         2. Different feature was standardized (not log returns)")
                    logger.error(f"         3. Statistics were computed differently during preprocessing")
                    logger.error(f"      Solution: Bypass destandardization by computing log returns directly from prices")
            else:
                logger.warning(f"   ⚠️  Shape mismatch, skipping validation")
        
        if close_price is not None and close_price_y is not None and close_price.dim() == 3 and close_price_y.dim() == 3:
            logger.info(f"✨ Computing CORRECT log returns directly from prices (bypassing standardization)...")
            
            # Ensure tensors are on same device as generated_samples
            device = generated_samples.device
            if close_price.device != device:
                close_price = close_price.to(device)
            if close_price_y.device != device:
                close_price_y = close_price_y.to(device)
            
            # close_price: [B_original*N, T_hist, 1], close_price_y: [B_original*N, T_future, 1]
            # We want to compute: log(price[t+1] / price[t]) * 100 for all timesteps
            # IMPORTANT: Dataset stores DailyLogReturn as PERCENTAGES (multiplied by 100)
            
            last_hist_price = close_price[:, -1:, :]  # [B_original*N, 1, 1]
            # Concatenate: [last_hist, future_0, future_1, ...]
            all_prices = torch.cat([last_hist_price, close_price_y], dim=1)  # [B_original*N, T_future+1, 1]
            
            # Compute log returns AS PERCENTAGES (multiply by 100)
            log_returns_correct = torch.log(all_prices[:, 1:, :] / all_prices[:, :-1, :]) * 100  # [B_original*N, T_future, 1]
            
            # Reshape to match real_samples structure: [B*n, T, N, F]
            # close_price has B_original*N entries, real_samples has [B*n, T, N, F]
            B_times_n = real_samples.shape[0]
            T = real_samples.shape[1]
            N = real_samples.shape[2]
            
            total_entries = log_returns_correct.shape[0]
            if total_entries == B_times_n * N and log_returns_correct.shape[1] == T:
                # Reshape [B*n*N, T, 1] -> [B*n, N, T, 1] -> [B*n, T, N, 1]
                log_returns_reshaped = log_returns_correct.view(B_times_n, N, T, 1).transpose(1, 2)  # [B*n, T, N, 1]
                
                logger.info(f"   ✅ Directly computed correct log returns from prices")
                logger.info(f"   Shape: {log_returns_reshaped.shape} (matches real_samples)")
                logger.info(f"   Stats: mean={log_returns_reshaped.mean().item():.6f}, std={log_returns_reshaped.std().item():.6f}")
                
                # Use these as the "destandardized" real samples
                real_destd = log_returns_reshaped
                use_direct_computation = True
            else:
                logger.warning(f"   ⚠️  Shape mismatch: close_price has {total_entries} entries, expected {B_times_n * N}")
                logger.warning(f"      Falling back to statistics-based destandardization")
                use_direct_computation = False
        else:
            logger.warning(f"   ⚠️  close_price or close_price_y not available")
            use_direct_computation = False
        
        if not use_direct_computation:
            # FALLBACK: Use statistics-based destandardization (original approach)
            logger.warning(f"   Using statistics-based destandardization (may be incorrect)")
            
            # Ensure consistent device
            device = generated_samples.device
            
            # Compute actual log returns from prices for ALL timesteps
            # close_price: [B_original*N, T_hist, 1], close_price_y: [B_original*N, T_future, 1]
            if close_price is not None and close_price_y is not None:
                if close_price.device != device:
                    close_price = close_price.to(device)
                if close_price_y.device != device:
                    close_price_y = close_price_y.to(device)
                
                last_hist_price = close_price[:, -1:, :]  # [B_original*N, 1, 1] - keep dim for broadcasting
                all_future_prices = close_price_y  # [B_original*N, T_future, 1]
                
                # Concatenate to get all transitions: [hist[-1], future[0], future[1], ...]
                prices_concat = torch.cat([last_hist_price, all_future_prices], dim=1)  # [B_original*N, T_future+1, 1]
                
                # Compute log returns AS PERCENTAGES: log(price[t+1] / price[t]) * 100
                log_returns_actual = torch.log(prices_concat[:, 1:, :] / prices_concat[:, :-1, :]) * 100  # [B_original*N, T_future, 1]
            
            # Reshape to [B_original, N, T_future, 1] assuming B_original*N entries
            # We need to figure out B_original and N
            # From stocks_index: [B*n, N] we know N
            # close_price has B_original*N entries where B_original might be B*n
            total_entries = log_returns_actual.shape[0]
            if total_entries % num_stocks == 0:
                B_original_inferred = total_entries // num_stocks
                log_returns_reshaped = log_returns_actual.view(B_original_inferred, num_stocks, -1, 1)  # [B_original, N, T, 1]
                
                # Compute per-stock statistics over time dimension
                # Take mean over (batch, time) to get per-stock stats
                mean_per_stock = log_returns_reshaped.mean(dim=(0, 2))  # [N, 1]
                std_per_stock = log_returns_reshaped.std(dim=(0, 2))  # [N, 1]
                
                mean_selected = mean_per_stock.squeeze(-1)  # [N,]
                std_selected = std_per_stock.squeeze(-1)  # [N,]
                
                logger.info(f"   ✅ Computed statistics from {B_original_inferred} batches × {num_stocks} stocks × {log_returns_reshaped.shape[2]} timesteps")
                logger.info(f"   Computed std range: [{std_selected.min().item():.6f}, {std_selected.max().item():.6f}]")
                logger.info(f"   Computed mean range: [{mean_selected.min().item():.6f}, {mean_selected.max().item():.6f}]")
                logger.info(f"   Compare with stored stats - std: [{std_array[stock_indices].min().item():.6f}, {std_array[stock_indices].max().item():.6f}]")
                logger.info(f"   Compare with stored stats - mean: [{mean_array[stock_indices].min().item():.6f}, {mean_array[stock_indices].max().item():.6f}]")
                
                use_computed_stats = True
            else:
                logger.warning(f"   ⚠️  Could not infer batch structure from close_price shape. Falling back to stored stats.")
                use_computed_stats = False
                mean_selected = mean_array[stock_indices]  # [N,] CPU tensor
                std_selected = std_array[stock_indices]    # [N,] CPU tensor
        else:
            logger.warning(f"   ⚠️  close_price or close_price_y not available. Using stored statistics (may be incorrect!).")
            use_computed_stats = False
            mean_selected = mean_array[stock_indices]  # [N,] CPU tensor
            std_selected = std_array[stock_indices]    # [N,] CPU tensor
        
        if not use_computed_stats:
            # Select per-stock stats using indices (FALLBACK - may be wrong!)
            mean_selected = mean_array[stock_indices]  # [N,] CPU tensor
            std_selected = std_array[stock_indices]    # [N,] CPU tensor
        
        # Move to GPU device and apply statistics-based destandardization (if needed)
        device = generated_samples.device
        
        if not use_direct_computation:
            mean_selected = mean_selected.to(device)
            std_selected = std_selected.to(device)
            
            # Reshape for broadcasting: [N,] -> [1, 1, N, 1] for samples [B*n, T, N, F]
            mean_broadcasted = mean_selected.view(1, 1, num_stocks, 1)
            std_broadcasted = std_selected.view(1, 1, num_stocks, 1)
            
            logger.info(f"   Broadcasting shape: mean/std [1, 1, {num_stocks}, 1] for samples [B*n={batch_size}, T={metadata['num_timesteps']}, N={num_stocks}, F={metadata['num_features']}]")
            
            # Log BEFORE destandardization
            logger.info(f"   BEFORE destandardization:")
            logger.info(f"      generated: mean={generated_samples.mean().item():.6f}, std={generated_samples.std().item():.6f}")
            logger.info(f"      real:      mean={real_samples.mean().item():.6f}, std={real_samples.std().item():.6f}")
            
            # Destandardize: x = z * std + mean
            generated_destd = generated_samples * std_broadcasted + mean_broadcasted  # [B*n, T, N, F]
            if 'real_destd' not in locals():  # Only destandardize if not already computed directly
                real_destd = real_samples * std_broadcasted + mean_broadcasted            # [B*n, T, N, F]
            
            # Log AFTER destandardization
            logger.info(f"   AFTER destandardization:")
            logger.info(f"      generated: mean={generated_destd.mean().item():.6f}, std={generated_destd.std().item():.6f}")
            logger.info(f"      real:      mean={real_destd.mean().item():.6f}, std={real_destd.std().item():.6f}")
        else:
            # real_destd already computed directly from prices, just destandardize generated_samples
            # For generated, we don't have prices, so we must use statistics
            logger.info(f"   Destandardizing generated samples using statistics...")
            mean_selected = mean_selected.to(device)
            std_selected = std_selected.to(device)
            mean_broadcasted = mean_selected.view(1, 1, num_stocks, 1)
            std_broadcasted = std_selected.view(1, 1, num_stocks, 1)
            generated_destd = generated_samples * std_broadcasted + mean_broadcasted
        
        # VALIDATION: Check if destandardized values match expected log returns from close_price
        close_price = metadata.get('close_price')
        close_price_y = metadata.get('close_price_y')
        
        logger.info(f"   VALIDATION CHECK:")
        logger.info(f"      close_price shape: {close_price.shape if close_price is not None else None}")
        logger.info(f"      close_price_y shape: {close_price_y.shape if close_price_y is not None else None}")
        logger.info(f"      real_destd shape: {real_destd.shape}")
        
        if close_price is not None and close_price_y is not None:
            # Ensure device consistency
            device = real_destd.device
            if close_price.device != device:
                close_price = close_price.to(device)
            if close_price_y.device != device:
                close_price_y = close_price_y.to(device)
            try:
                # Compute expected log returns from prices for ALL timesteps
                if close_price.dim() == 3 and close_price_y.dim() == 3:  # [B_original*N, T_hist, 1] and [B_original*N, T_future, 1]
                    # Get last historical price
                    last_hist_price = close_price[:, -1, :]  # [B_original*N, 1]
                    
                    # Compute log returns AS PERCENTAGES for all future timesteps
                    # price[0] comes from log(close_price_y[0] / close_price[-1]) * 100
                    first_future_price = close_price_y[:, 0, :]
                    expected_lr_0 = torch.log(first_future_price / last_hist_price) * 100  # [B_original*N, 1]
                    
                    # real_destd: [B*n, T, N, F] - reshape to get first timestep
                    B_times_n = real_destd.shape[0]
                    N = real_destd.shape[2]
                    actual_lr_0 = real_destd[:, 0, :, 0]  # [B*n, N]
                    actual_lr_flat = actual_lr_0.reshape(-1, 1)  # [B*n*N, 1]
                    
                    # Take first entries to match expected_lr size
                    min_size = min(expected_lr_0.shape[0], actual_lr_flat.shape[0])
                    expected_lr_subset = expected_lr_0[:min_size]
                    actual_lr_subset = actual_lr_flat[:min_size]
                    
                    # Compute difference
                    diff = (expected_lr_subset - actual_lr_subset).abs().mean().item()
                    rel_diff = diff / expected_lr_subset.abs().mean().item() * 100 if expected_lr_subset.abs().mean().item() > 1e-6 else 0
                    
                    logger.info(f"      DESTANDARDIZATION VALIDATION (first timestep):")
                    logger.info(f"         Expected LR from prices: mean={expected_lr_subset.mean().item():.6f}, std={expected_lr_subset.std().item():.6f}, min={expected_lr_subset.min().item():.6f}, max={expected_lr_subset.max().item():.6f}")
                    logger.info(f"         Actual destd LR:        mean={actual_lr_subset.mean().item():.6f}, std={actual_lr_subset.std().item():.6f}, min={actual_lr_subset.min().item():.6f}, max={actual_lr_subset.max().item():.6f}")
                    logger.info(f"         Abs diff: {diff:.6f}, Rel diff: {rel_diff:.2f}%")
                    
                    if rel_diff < 5:
                        logger.info(f"         ✅ Destandardization appears reasonable (rel diff < 5%)")
                    elif rel_diff < 20:
                        logger.warning(f"         ⚠️  Moderate error in destandardization (5% < rel diff < 20%)")
                    else:
                        logger.error(f"         ❌ LARGE ERROR! Destandardization is using WRONG statistics (rel diff > 20%)")
                        logger.error(f"            The per-stock mean/std in dataset_info do NOT match the actual data!")
                        logger.error(f"            Expected std ~{expected_lr_subset.std().item():.6f} but destandardization gives std ~{actual_lr_subset.std().item():.6f}")
            except Exception as e:
                logger.warning(f"      Validation failed with error: {e}")
        
        # Store per-stock mean for direction accuracy (de-meaned sign test)
        # mean_broadcasted is [1, 1, N, 1] matching destandardised return space
        metadata['per_stock_mean'] = mean_broadcasted  # [1, 1, N, 1]
        
        logger.info(f"   ✅ Destandardization complete")
        
        return generated_destd, real_destd
    
    def _convert_log_returns_to_prices_v2(
        self,
        pred_log_returns: torch.Tensor,
        target_log_returns: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert destandardized log returns to actual closing prices.
        Follows the same pattern as task v1.
        
        Args:
            pred_log_returns: [B*n, T, N, F] or [B, n, T, N, F] predicted log returns (DESTANDARDIZED, as PERCENTAGES)
            target_log_returns: [B*n, T, N, F] or [B, n, T, N, F] target log returns (DESTANDARDIZED, as PERCENTAGES)
            metadata: Dict containing:
                - 'close_price': [B*N, T_hist, 1] or [B, T_hist, N, 1] historical closing prices
                - 'batch_size': B
                - 'num_stocks': N
        
        Returns:
            Tuple of (pred_prices, target_prices) with same shape as inputs
        """
        close_price = metadata.get('close_price')
        if close_price is None:
            logger.warning("⚠️  No close_price in metadata. Cannot convert to prices.")
            return None, None
        
        # CRITICAL: Destandardized log returns are in PERCENTAGES (multiplied by 100 during preprocessing)
        # Need to divide by 100 to get actual log return ratios for price conversion
        # Original preprocessing: DailyLogReturn = np.log(close[t] / close[t-1]) * 100
        # For price conversion: price[t] = price[t-1] * exp(log_return / 100)
        pred_log_returns_ratio = pred_log_returns / 100.0
        target_log_returns_ratio = target_log_returns / 100.0
        
        # Get last historical closing price
        if close_price.dim() == 3:  # [B_original*N, T_hist, 1] where B_original = B*n
            last_prices = close_price[:, -1, :]  # [B_original*N, 1]
            # Infer B_original from the actual tensor size instead of metadata
            # to handle both ReplicatedDataset (correctly batched) and
            # GRW-style replication (batch-aware repeat_interleave).
            N = metadata.get('num_stocks', 1)
            B_original = close_price.shape[0] // N
            last_prices = last_prices.view(B_original, N, 1)  # [B_original, N, 1]
        elif close_price.dim() == 4:  # [B_original, T_hist, N, 1]
            last_prices = close_price[:, -1, :, :]  # [B_original, N, 1]
        else:
            logger.error(f"Unexpected close_price shape: {close_price.shape}")
            return None, None
        
        # Clamp log returns to prevent exp() overflow
        # 1.0 = exp(1) ≈ 2.72x (172% gain) or exp(-1) ≈ 0.37x (63% loss) per day
        # Handles extreme events (crashes, circuit breakers) while preventing numerical issues
        LOG_RETURN_CLAMP = 1.0
        pred_clamped = torch.clamp(pred_log_returns_ratio, -LOG_RETURN_CLAMP, LOG_RETURN_CLAMP)
        target_clamped = torch.clamp(target_log_returns_ratio, -LOG_RETURN_CLAMP, LOG_RETURN_CLAMP)
        
        # Check if clamping was significant
        pred_clamped_pct = (pred_log_returns_ratio.abs() > LOG_RETURN_CLAMP).float().mean().item() * 100
        target_clamped_pct = (target_log_returns_ratio.abs() > LOG_RETURN_CLAMP).float().mean().item() * 100
        if pred_clamped_pct > 1 or target_clamped_pct > 1:
            logger.warning(f"⚠️  Clamped {pred_clamped_pct:.1f}% pred, {target_clamped_pct:.1f}% target to ±{LOG_RETURN_CLAMP}")
            # Show statistics to help diagnose
            logger.warning(f"   Destandardized log return stats (as ratios, after /100):")
            logger.warning(f"     pred: min={pred_log_returns_ratio.min().item():.4f}, max={pred_log_returns_ratio.max().item():.4f}, std={pred_log_returns_ratio.std().item():.4f}")
            logger.warning(f"     target: min={target_log_returns_ratio.min().item():.4f}, max={target_log_returns_ratio.max().item():.4f}, std={target_log_returns_ratio.std().item():.4f}")
            logger.warning(f"   Note: Daily log returns |r| > 1.0 mean >170% single-day moves (extremely rare)")
        
        # Convert to prices: price_t = last_price * exp(cumsum(log_returns))
        if pred_log_returns.dim() == 4:  # [B, T, N, F] - after ensemble mean
            # Cumulative sum over time
            cumsum_pred = torch.cumsum(pred_clamped, dim=1)  # [B, T, N, F]
            cumsum_target = torch.cumsum(target_clamped, dim=1)  # [B, T, N, F]
            
            # last_prices is [B_original, N, 1] but log returns are [B, T, N, F]
            # If B_original > B, subsample (due to ReplicatedDataset, all n copies have same price)
            B_actual = pred_log_returns.shape[0]
            if last_prices.shape[0] > B_actual:
                n = metadata.get('n_samples_per_input', 1)
                last_prices_B = last_prices[::n, :, :]  # [B, N, 1]
            else:
                last_prices_B = last_prices  # [B, N, 1]
            
            # Expand time dimension: [B, N, 1] -> [B, 1, N, 1]
            last_prices_expanded = last_prices_B.unsqueeze(1)  # [B, 1, N, 1]
            
            pred_prices = last_prices_expanded * torch.exp(cumsum_pred)
            target_prices = last_prices_expanded * torch.exp(cumsum_target)
            
        elif pred_log_returns.dim() == 5:  # [B, n, T, N, F] ensemble format
            # Cumulative sum over time (dim=2)
            cumsum_pred = torch.cumsum(pred_clamped, dim=2)  # [B, n, T, N, F]
            cumsum_target = torch.cumsum(target_clamped, dim=2)  # [B, n, T, N, F]
            
            # last_prices is [B_original, N, 1] but we need [B, N, 1] for ensemble
            # Since B_original = B*n, we need to reshape: take first B entries
            # Actually, all B*n entries should have same price (replicated), so just take first B
            n = metadata.get('n_samples_per_input', 1)
            B = pred_log_returns.shape[0]  # True batch size from ensemble tensor
            N = last_prices.shape[1]
            
            # Check if subsampling is needed (similar to 4D case)
            if last_prices.shape[0] > B:
                # Reshape from [B*n, N, 1] to [B, n, N, 1] and take first along n
                # Or simply take every n-th entry
                last_prices_B = last_prices[::n, :, :]  # [B, N, 1]
            else:
                last_prices_B = last_prices  # [B, N, 1] - already correct size
            
            # Expand: [B, N, 1] -> [B, 1, 1, N, 1]
            last_prices_expanded = last_prices_B.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 1]
            
            pred_prices = last_prices_expanded * torch.exp(cumsum_pred)
            target_prices = last_prices_expanded * torch.exp(cumsum_target)
        else:
            logger.error(f"Unexpected pred_log_returns shape: {pred_log_returns.shape}")
            return None, None
        
        logger.info(f"💰 Converted log returns to prices: pred {pred_prices.shape}, target {target_prices.shape}")
        logger.info(f"   Price stats - pred: mean={pred_prices.mean().item():.2f}, std={pred_prices.std().item():.2f}")
        logger.info(f"   Price stats - target: mean={target_prices.mean().item():.2f}, std={target_prices.std().item():.2f}")
        
        # VALIDATION: Check if converted TARGET prices match close_price_y (ground truth)
        close_price_y = metadata.get('close_price_y')
        if close_price_y is not None and target_prices.dim() == 4:
            # Ensure close_price_y is on the same device as target_prices
            if close_price_y.device != target_prices.device:
                close_price_y = close_price_y.to(target_prices.device)
            
            # Get first timestep of TARGET prices (should match close_price_y[0])
            first_target_price = target_prices[:, 0, :, 0]  # [B, N]
            
            # Get first timestep from close_price_y
            if close_price_y.dim() == 3:  # [B_original*N, T, 1]
                B_actual = target_prices.shape[0]
                N = target_prices.shape[2]
                # Infer how many batch items are in close_price_y and
                # subsample to match the ensemble-mean batch size.
                B_cpy = close_price_y.shape[0] // N
                if B_cpy > B_actual:
                    n = metadata.get('n_samples_per_input', 1)
                    # Take one representative per unique input
                    cpy_reshaped = close_price_y[:B_cpy * N, 0, 0].view(B_cpy, N)
                    first_expected_price = cpy_reshaped[::n, :]  # [B_actual, N]
                else:
                    first_expected_price = close_price_y[:B_actual*N, 0, 0].view(B_actual, N)
            elif close_price_y.dim() == 4:  # [B, T, N, 1]
                first_expected_price = close_price_y[:, 0, :, 0]  # [B, N]
            else:
                first_expected_price = None
            
            if first_expected_price is not None and first_target_price.shape == first_expected_price.shape:
                price_diff = (first_target_price - first_expected_price).abs().mean().item()
                price_rel_diff = price_diff / first_expected_price.mean().item() * 100
                logger.info(f"   VALIDATION: First TARGET price vs close_price_y[0]:")
                logger.info(f"      Expected (close_price_y[0]): mean={first_expected_price.mean().item():.2f}")
                logger.info(f"      Actual (converted target):   mean={first_target_price.mean().item():.2f}")
                logger.info(f"      Abs diff: {price_diff:.2f}, Rel diff: {price_rel_diff:.2f}%")
                if price_rel_diff < 0.1:
                    logger.info(f"      ✅ Price conversion is CORRECT (rel diff < 0.1%)")
                elif price_rel_diff < 1:
                    logger.warning(f"      ⚠️  Small error in price conversion (rel diff < 1%)")
                else:
                    logger.error(f"      ❌ LARGE ERROR! Price conversion or destandardization is WRONG")
                    logger.error(f"         Either: 1) Destandardization used wrong stats, or")
                    logger.error(f"                 2) Log returns don't represent log(price[t+1]/price[t])")
        
        return pred_prices, target_prices
    
    def _process_ensemble_v2(
        self,
        generated_destd: torch.Tensor,
        real_destd: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reshape destandardized samples to ensemble format and compute diversity.
        
        Args:
            generated_destd: Destandardized generated samples [B*n, T, N, F]
            real_destd: Destandardized real samples [B*n, T, N, F]
            metadata: Dict with n_samples_per_input
        
        Returns:
            Dict with diversity metrics
        
        Modifies metadata in-place:
            - 'gen_ensemble': [B, n, T, N, F]
            - 'target_ensemble': [B, n, T, N, F]
            - 'target_single': [B, T, N, F] — first real replica (true observation for CRPS)
            - 'pred_mean': [B, T, N, F] ensemble mean of generated
            - 'target_mean': [B, T, N, F] ensemble mean of real (kept for viz backward compat)
        """
        n = metadata.get('n_samples_per_input')
        if n is None:
            raise ValueError("metadata['n_samples_per_input'] is required")
        
        batch_size = metadata['batch_size']
        if batch_size % n != 0:
            raise ValueError(
                f"batch_size={batch_size} is not divisible by n_samples_per_input={n}"
            )
        
        B = batch_size // n
        T = metadata['num_timesteps']
        N = metadata['num_stocks']
        F = metadata['num_features']
        
        # Reshape to ensemble format [B, n, T, N, F]
        generated_ensemble = generated_destd.view(B, n, T, N, F)
        real_ensemble = real_destd.view(B, n, T, N, F)
        
        logger.info(f"📦 Reshaped to ensemble format: [B*n={batch_size}, T, N, F] -> [B={B}, n={n}, T={T}, N={N}, F={F}]")
        
        # Compute ensemble diversity (std across generated members, dim=1 = n)
        if n > 1:
            sample_std = generated_ensemble.std(dim=1).mean().item()
            diversity = sample_std
            logger.info(f"📊 Ensemble diversity (std across n): {diversity:.6f}")
        else:
            diversity = 0.0
            logger.info(f"📊 Single sample (n=1), diversity=0")
        
        # Ensemble mean of generated predictions [B, T, N, F]
        pred_mean = generated_ensemble.mean(dim=1)  # [B, T, N, F]  — mean over n (dim=1)
        
        # Single ground-truth observation: first replica (all n replicas are identical
        # for ReplicatedDataset, but taking [:, 0] is semantically correct for CRPS
        # regardless of whether replicas carry augmentation noise in the future)
        target_single = real_ensemble[:, 0]  # [B, T, N, F]  — index dim=1 (n)
        
        # Keep target_mean for backward compatibility (viz reads it)
        target_mean = real_ensemble.mean(dim=1)  # [B, T, N, F]
        
        logger.info(f"📈 Ensemble mean: pred [B={B}, T={T}, N={N}, F={F}], "
                     f"target_single [B={B}, T={T}, N={N}, F={F}]")
        
        # Store in metadata for downstream use
        metadata['gen_ensemble'] = generated_ensemble      # [B, n, T, N, F]
        metadata['target_ensemble'] = real_ensemble         # [B, n, T, N, F]
        metadata['target_single'] = target_single           # [B, T, N, F]
        metadata['pred_mean'] = pred_mean                   # [B, T, N, F]
        metadata['target_mean'] = target_mean               # [B, T, N, F] (for viz)
        
        # Update batch_size to reflect the true batch size (not B*n)
        metadata['batch_size_original'] = metadata['batch_size']  # Save original for reference
        metadata['batch_size'] = B  # Update to true batch size after ensemble split
        
        return {'sample_diversity': diversity}
    
    def _compute_three_tier_metrics(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute three tiers of evaluation metrics.
        
        Tier 1 — Primary (daily log returns): MAE, RMSE, MAPE, sMAPE, direction accuracy,
                 correlation, CRPS, MIS, coverage, spread.
        Tier 2 — Secondary (closing prices): MAE, RMSE, MSE, CRPS, MIS, coverage, spread,
                 plus per-horizon temporal metrics.
        Tier 3 — Cumulative (final-horizon log return): MAE, RMSE, direction accuracy,
                 CRPS, MIS, coverage.
        
        Dimension conventions (see docstrings of upstream methods for derivation):
            B = true batch size (after ensemble split)
            n = number of ensemble members (n_samples_per_input)
            T = forecast horizon (num_timesteps)
            N = number of stocks (num_stocks)
            F = number of features (num_features)
        
        Args:
            metadata: Dict containing (all set by evaluate_samples before this call):
                - 'pred_mean':           [B, T, N, F]      ensemble mean of generated log returns
                - 'target_single':       [B, T, N, F]      single ground-truth log returns
                - 'gen_ensemble':        [B, n, T, N, F]   full generated ensemble log returns
                - 'pred_prices':         [B, T, N, F]      mean of ensemble prices
                - 'target_prices':       [B, T, N, F]      ground-truth closing prices
                - 'gen_ensemble_prices': [B, n, T, N, F]   full generated ensemble prices
                - 'per_stock_mean':      [1, 1, N, 1]      per-stock training mean mu_s (for dir acc)
        
        Returns:
            Dictionary of all metrics with appropriate prefixes and legacy aliases.
        """
        from graph_signal_diffusion.evaluation.probabilistic_metrics import compute_probabilistic_metrics
        from graph_signal_diffusion.evaluation.metrics import compute_mape, compute_smape
        
        metrics: Dict[str, float] = {}
        
        pred_mean = metadata['pred_mean']                          # [B, T, N, F]
        target_single = metadata['target_single']                  # [B, T, N, F]
        gen_ensemble = metadata.get('gen_ensemble')                # [B, n, T, N, F]
        pred_prices = metadata.get('pred_prices')                  # [B, T, N, F]
        target_prices = metadata.get('target_prices')              # [B, T, N, F]
        gen_ensemble_prices = metadata.get('gen_ensemble_prices')  # [B, n, T, N, F]
        per_stock_mean = metadata.get('per_stock_mean')            # [1, 1, N, 1] or None
        
        B, T, N, F = pred_mean.shape
        
        # =====================================================================
        # TIER 1: Primary metrics on daily log returns
        # =====================================================================
        logger.info("━━━ Tier 1: Primary metrics (daily log returns) ━━━")
        
        # --- Point metrics ---
        # pred_mean: [B, T, N, F],  target_single: [B, T, N, F]
        return_mse = torch.nn.functional.mse_loss(pred_mean, target_single).item()
        return_mae = torch.nn.functional.l1_loss(pred_mean, target_single).item()
        return_rmse = (return_mse ** 0.5)
        return_mape = compute_mape(pred_mean, target_single)
        return_smape = compute_smape(pred_mean, target_single)
        
        metrics['return_mse'] = return_mse
        metrics['return_mae'] = return_mae
        metrics['return_rmse'] = return_rmse
        metrics['return_mape'] = return_mape
        metrics['return_smape'] = return_smape
        
        logger.info(f"   MAE={return_mae:.6f}, RMSE={return_rmse:.6f}, "
                     f"MAPE={return_mape:.2f}%, sMAPE={return_smape:.2f}%")
        
        # --- Direction accuracy ---
        # Two thresholds reported per day:
        #   * vs per-stock long-run mean μ_s — removes the positive drift bias
        #     (most stocks have μ_s > 0); answers "did the day beat the stock's
        #     own long-run drift?". Logged as ``direction_accuracy`` and
        #     ``direction_accuracy_majority_vote``.
        #   * vs absolute zero — sign test "did the stock go up?". Logged as
        #     ``direction_accuracy_absolute`` and
        #     ``direction_accuracy_majority_vote_absolute``.
        # Both share a helper (_compute_direction_accuracy_pair) that handles
        # ensemble (per-sample + majority-vote) and single-sample paths.
        if per_stock_mean is None:
            logger.warning("⚠️  per_stock_mean not available, direction_accuracy uses threshold=0 (biased)")
        threshold_mu = per_stock_mean if per_stock_mean is not None else 0.0

        direction_acc, majority_acc = _compute_direction_accuracy_pair(
            target=target_single,
            gen_ensemble=gen_ensemble,
            pred_mean=pred_mean,
            threshold=threshold_mu,
            ensemble_dim=1,
        )
        metrics['direction_accuracy'] = direction_acc
        if majority_acc is not None:
            metrics['direction_accuracy_majority_vote'] = majority_acc
            logger.info(f"   Direction accuracy vs μ_s (per-sample): {direction_acc:.4f}, "
                         f"majority-vote: {majority_acc:.4f}")
        else:
            logger.info(f"   Direction accuracy vs μ_s (single-sample): {direction_acc:.4f}")

        abs_direction_acc, abs_majority_acc = _compute_direction_accuracy_pair(
            target=target_single,
            gen_ensemble=gen_ensemble,
            pred_mean=pred_mean,
            threshold=0.0,
            ensemble_dim=1,
        )
        metrics['direction_accuracy_absolute'] = abs_direction_acc
        if abs_majority_acc is not None:
            metrics['direction_accuracy_majority_vote_absolute'] = abs_majority_acc
            logger.info(f"   Direction accuracy vs 0   (per-sample): {abs_direction_acc:.4f}, "
                         f"majority-vote: {abs_majority_acc:.4f}")
        else:
            logger.info(f"   Direction accuracy vs 0   (single-sample): {abs_direction_acc:.4f}")
        
        # --- Structural metrics (temporal + distributional) ---
        gen_tensor = gen_ensemble if gen_ensemble is not None else pred_mean

        metrics["volclustering_gen"] = self._lag1_acf(gen_tensor, squared=True)
        metrics["volclustering_real"] = self._lag1_acf(target_single, squared=True)
        metrics["volclustering_gap"] = abs(metrics["volclustering_gen"] - metrics["volclustering_real"])

        metrics["momentum_gen"] = self._lag1_acf(gen_tensor, squared=False)
        metrics["momentum_real"] = self._lag1_acf(target_single, squared=False)
        metrics["momentum_gap"] = abs(metrics["momentum_gen"] - metrics["momentum_real"])

        metrics["kurtosis_gen"] = self._excess_kurtosis(gen_tensor)
        metrics["kurtosis_real"] = self._excess_kurtosis(target_single)
        metrics["kurtosis_gap"] = abs(metrics["kurtosis_gen"] - metrics["kurtosis_real"])

        ev_gen = self._top1_eigenvalue(gen_tensor)
        ev_real = self._top1_eigenvalue(target_single)
        if ev_gen is not None:
            metrics["eigval1_gen"] = ev_gen
        if ev_real is not None:
            metrics["eigval1_real"] = ev_real
        if ev_gen is not None and ev_real is not None and ev_real >= 0.01:
            metrics["eigval1_ratio"] = ev_gen / ev_real

        logger.info(f"   VolCluster: gen={metrics['volclustering_gen']:.4f}, "
                    f"real={metrics['volclustering_real']:.4f}, "
                    f"gap={metrics['volclustering_gap']:.4f}")
        logger.info(f"   Momentum:   gen={metrics['momentum_gen']:.4f}, "
                    f"real={metrics['momentum_real']:.4f}")
        logger.info(f"   Kurtosis:   gen={metrics['kurtosis_gen']:.2f}, "
                    f"real={metrics['kurtosis_real']:.2f}")
        
        # --- Probabilistic metrics on log returns ---
        # gen_ensemble: [B, n, T, N, F],  target_single: [B, T, N, F]
        # compute_probabilistic_metrics unsqueezes target at ensemble_dim=1 internally.
        if gen_ensemble is not None and gen_ensemble.shape[1] > 1:
            prob_ret = compute_probabilistic_metrics(
                predictions=gen_ensemble,       # [B, n, T, N, F]
                target=target_single,           # [B, T, N, F]
                ensemble_dim=1,                 # n is at dim 1
                alphas=[0.01, 0.05, 0.1, 0.2],  # 99 / 95 / 90 / 80% intervals
            )
            metrics['return_crps'] = prob_ret.get('crps_mean', 0.0)
            metrics['return_crps_std'] = prob_ret.get('crps_std', 0.0)
            metrics['return_mis_99'] = prob_ret.get('mis_99', 0.0)
            metrics['return_mis_95'] = prob_ret.get('mis_95', 0.0)
            metrics['return_mis_90'] = prob_ret.get('mis_90', 0.0)
            metrics['return_mis_80'] = prob_ret.get('mis_80', 0.0)
            metrics['return_coverage_99'] = prob_ret.get('coverage_99', 0.0)
            metrics['return_coverage_95'] = prob_ret.get('coverage_95', 0.0)
            metrics['return_coverage_90'] = prob_ret.get('coverage_90', 0.0)
            metrics['return_coverage_80'] = prob_ret.get('coverage_80', 0.0)
            metrics['return_spread'] = prob_ret.get('ensemble_spread_mean', 0.0)
            metrics['return_spread_std'] = prob_ret.get('ensemble_spread_std', 0.0)
            logger.info(f"   CRPS={metrics['return_crps']:.6f}, "
                         f"MIS90={metrics['return_mis_90']:.6f}, "
                         f"Cov90={metrics['return_coverage_90']:.4f}, "
                         f"Cov95={metrics['return_coverage_95']:.4f}, "
                         f"Cov99={metrics['return_coverage_99']:.4f}")
        
        # --- Legacy aliases (point to primary return metrics for backward compatibility) ---
        metrics['mae'] = return_mae
        metrics['rmse'] = return_rmse
        metrics['mse'] = return_mse
        metrics['crps_mean'] = metrics.get('return_crps', 0.0)
        metrics['crps'] = metrics.get('return_crps', 0.0)
        metrics['crps_std'] = metrics.get('return_crps_std', 0.0)
        metrics['mis_99'] = metrics.get('return_mis_99', 0.0)
        metrics['mis_95'] = metrics.get('return_mis_95', 0.0)
        metrics['mis_90'] = metrics.get('return_mis_90', 0.0)
        metrics['mis_80'] = metrics.get('return_mis_80', 0.0)
        metrics['coverage_99'] = metrics.get('return_coverage_99', 0.0)
        metrics['coverage_95'] = metrics.get('return_coverage_95', 0.0)
        metrics['coverage_90'] = metrics.get('return_coverage_90', 0.0)
        metrics['coverage_80'] = metrics.get('return_coverage_80', 0.0)
        metrics['ensemble_spread_mean'] = metrics.get('return_spread', 0.0)
        metrics['ensemble_spread_std'] = metrics.get('return_spread_std', 0.0)
        
        # =====================================================================
        # TIER 2: Secondary metrics on closing prices
        # =====================================================================
        if pred_prices is not None and target_prices is not None:
            logger.info("━━━ Tier 2: Secondary metrics (closing prices) ━━━")
            
            # pred_prices: [B, T, N, F],  target_prices: [B, T, N, F]
            price_mse = torch.nn.functional.mse_loss(pred_prices, target_prices).item()
            price_mae = torch.nn.functional.l1_loss(pred_prices, target_prices).item()
            price_rmse = (price_mse ** 0.5)
            
            metrics['price_mse'] = price_mse
            metrics['price_mae'] = price_mae
            metrics['price_rmse'] = price_rmse
            
            logger.info(f"   MAE=${price_mae:.2f}, RMSE=${price_rmse:.2f}")
            
            # Probabilistic metrics on prices
            # gen_ensemble_prices: [B, n, T, N, F],  target_prices: [B, T, N, F]
            if gen_ensemble_prices is not None and gen_ensemble_prices.shape[1] > 1:
                prob_price = compute_probabilistic_metrics(
                    predictions=gen_ensemble_prices,  # [B, n, T, N, F]
                    target=target_prices,             # [B, T, N, F]
                    ensemble_dim=1,                   # n is at dim 1
                    alphas=[0.01, 0.05, 0.1, 0.2],    # 99 / 95 / 90 / 80% intervals
                )
                metrics['price_crps'] = prob_price.get('crps_mean', 0.0)
                metrics['price_crps_std'] = prob_price.get('crps_std', 0.0)
                metrics['price_mis_99'] = prob_price.get('mis_99', 0.0)
                metrics['price_mis_95'] = prob_price.get('mis_95', 0.0)
                metrics['price_mis_90'] = prob_price.get('mis_90', 0.0)
                metrics['price_mis_80'] = prob_price.get('mis_80', 0.0)
                metrics['price_coverage_99'] = prob_price.get('coverage_99', 0.0)
                metrics['price_coverage_95'] = prob_price.get('coverage_95', 0.0)
                metrics['price_coverage_90'] = prob_price.get('coverage_90', 0.0)
                metrics['price_coverage_80'] = prob_price.get('coverage_80', 0.0)
                metrics['price_spread'] = prob_price.get('ensemble_spread_mean', 0.0)
                logger.info(f"   CRPS=${metrics['price_crps']:.4f}, "
                             f"MIS90=${metrics['price_mis_90']:.2f}")
            
            # Per-horizon temporal metrics on prices
            # Index dim 1 (T) for each horizon; dims 2,3 = N, F
            for t in range(T):
                t_mse = torch.nn.functional.mse_loss(
                    pred_prices[:, t, :, :], target_prices[:, t, :, :]
                ).item()
                t_mae = torch.nn.functional.l1_loss(
                    pred_prices[:, t, :, :], target_prices[:, t, :, :]
                ).item()
                metrics[f'price_mse_t{t+1}'] = t_mse
                metrics[f'price_mae_t{t+1}'] = t_mae
            
            # Short/medium/long term aggregation
            if T >= 3:
                short_term = slice(0, T // 3)
                medium_term = slice(T // 3, 2 * T // 3)
                long_term = slice(2 * T // 3, T)
                metrics['price_mse_short_term'] = torch.nn.functional.mse_loss(
                    pred_prices[:, short_term], target_prices[:, short_term]
                ).item()
                metrics['price_mse_medium_term'] = torch.nn.functional.mse_loss(
                    pred_prices[:, medium_term], target_prices[:, medium_term]
                ).item()
                metrics['price_mse_long_term'] = torch.nn.functional.mse_loss(
                    pred_prices[:, long_term], target_prices[:, long_term]
                ).item()
        else:
            logger.warning("⚠️  No price data available, skipping Tier 2 (price) metrics")
        
        # =====================================================================
        # TIER 3: Cumulative return metrics at final horizon t=T
        # =====================================================================
        logger.info("━━━ Tier 3: Cumulative return metrics (final horizon) ━━━")
        
        # Cumulative log returns: R_{b,t,s} = sum_{tau=1}^{t} r_{b,tau,s}
        # pred_mean: [B, T, N, F] → cumsum over dim=1 (T)
        cum_pred = torch.cumsum(pred_mean, dim=1)         # [B, T, N, F]
        cum_target = torch.cumsum(target_single, dim=1)    # [B, T, N, F]
        
        # Extract final horizon t=T: slice dim=1 (T) at index -1 → [B, N, F]
        cum_pred_T = cum_pred[:, -1, :, :]       # [B, N, F]
        cum_target_T = cum_target[:, -1, :, :]    # [B, N, F]
        
        cum_T_mae = torch.nn.functional.l1_loss(cum_pred_T, cum_target_T).item()
        cum_T_mse = torch.nn.functional.mse_loss(cum_pred_T, cum_target_T).item()
        cum_T_rmse = (cum_T_mse ** 0.5)
        
        metrics['cum_T_mae'] = cum_T_mae
        metrics['cum_T_rmse'] = cum_T_rmse
        
        logger.info(f"   MAE={cum_T_mae:.6f}, RMSE={cum_T_rmse:.6f}")
        
        # --- Cumulative direction accuracy at T ---
        # Two thresholds, same structure as the per-day metric:
        #   * vs T·μ_s — "did the T-day return beat the stock's long-run expected
        #     T-day drift?". Logged as ``cum_T_direction_accuracy`` and
        #     ``cum_T_direction_accuracy_majority_vote``.
        #   * vs 0 — absolute sign test on the cumulative return. Logged as
        #     ``cum_T_direction_accuracy_absolute`` and
        #     ``cum_T_direction_accuracy_majority_vote_absolute``.
        if per_stock_mean is not None:
            cum_threshold_mu = (T * per_stock_mean.squeeze(0).squeeze(0))  # [N, 1]
        else:
            cum_threshold_mu = 0.0

        # Compute cumulative ensemble once — reused below for CRPS / coverage.
        if gen_ensemble is not None and gen_ensemble.shape[1] > 1:
            cum_ensemble = torch.cumsum(gen_ensemble, dim=2)  # [B, n, T, N, F]
            cum_ensemble_T = cum_ensemble[:, :, -1, :, :]     # [B, n, N, F]
        else:
            cum_ensemble = None
            cum_ensemble_T = None

        cum_T_dir_acc, cum_T_majority_acc = _compute_direction_accuracy_pair(
            target=cum_target_T,
            gen_ensemble=cum_ensemble_T,
            pred_mean=cum_pred_T,
            threshold=cum_threshold_mu,
            ensemble_dim=1,
        )
        metrics['cum_T_direction_accuracy'] = cum_T_dir_acc
        if cum_T_majority_acc is not None:
            metrics['cum_T_direction_accuracy_majority_vote'] = cum_T_majority_acc
            logger.info(f"   Cum-T direction accuracy vs T·μ_s (per-sample): {cum_T_dir_acc:.4f}, "
                         f"majority-vote: {cum_T_majority_acc:.4f}")
        else:
            logger.info(f"   Cum-T direction accuracy vs T·μ_s (single-sample): {cum_T_dir_acc:.4f}")

        abs_cum_T_dir_acc, abs_cum_T_majority_acc = _compute_direction_accuracy_pair(
            target=cum_target_T,
            gen_ensemble=cum_ensemble_T,
            pred_mean=cum_pred_T,
            threshold=0.0,
            ensemble_dim=1,
        )
        metrics['cum_T_direction_accuracy_absolute'] = abs_cum_T_dir_acc
        if abs_cum_T_majority_acc is not None:
            metrics['cum_T_direction_accuracy_majority_vote_absolute'] = abs_cum_T_majority_acc
            logger.info(f"   Cum-T direction accuracy vs 0     (per-sample): {abs_cum_T_dir_acc:.4f}, "
                         f"majority-vote: {abs_cum_T_majority_acc:.4f}")
        else:
            logger.info(f"   Cum-T direction accuracy vs 0     (single-sample): {abs_cum_T_dir_acc:.4f}")

        # Probabilistic metrics on cumulative ensemble returns at T
        if gen_ensemble is not None and gen_ensemble.shape[1] > 1:
            # cum_ensemble and cum_ensemble_T already computed above
            
            # CRPS target: cum_target_T [B, N, F]
            # compute_probabilistic_metrics will unsqueeze at ensemble_dim=1 internally.
            # cum_T probabilistic metrics — same 99/95/90/80 quantile grid.
            prob_cum = compute_probabilistic_metrics(
                predictions=cum_ensemble_T,   # [B, n, N, F]
                target=cum_target_T,          # [B, N, F]
                ensemble_dim=1,               # n is at dim 1
                alphas=[0.01, 0.05, 0.1, 0.2],
            )
            metrics['cum_T_crps'] = prob_cum.get('crps_mean', 0.0)
            metrics['cum_T_mis_99'] = prob_cum.get('mis_99', 0.0)
            metrics['cum_T_mis_95'] = prob_cum.get('mis_95', 0.0)
            metrics['cum_T_mis_90'] = prob_cum.get('mis_90', 0.0)
            metrics['cum_T_mis_80'] = prob_cum.get('mis_80', 0.0)
            metrics['cum_T_coverage_99'] = prob_cum.get('coverage_99', 0.0)
            metrics['cum_T_coverage_95'] = prob_cum.get('coverage_95', 0.0)
            metrics['cum_T_coverage_90'] = prob_cum.get('coverage_90', 0.0)
            metrics['cum_T_coverage_80'] = prob_cum.get('coverage_80', 0.0)
            logger.info(f"   CRPS={metrics['cum_T_crps']:.6f}, "
                         f"Cov90={metrics['cum_T_coverage_90']:.4f}, "
                         f"Cov95={metrics['cum_T_coverage_95']:.4f}, "
                         f"Cov99={metrics['cum_T_coverage_99']:.4f}")
        
        return metrics
    
    def _visualize_results_v2(self, metadata: Dict[str, Any], viz_save_dir: Optional[str] = None) -> None:
        """
        Visualize prediction results for TaskV2 with full plotting support.
        
        Creates time series plots of predicted vs actual log returns for selected stocks,
        including ensemble confidence bands when available.
        
        Args:
            metadata: Dict containing:
                - 'pred_mean': [B, T, N, F] ensemble mean predictions
                - 'target_mean': [B, T, N, F] ensemble mean targets
                - 'gen_ensemble': [B, n, T, N, F] full ensemble (optional)
                - 'stock_symbols': List of stock symbols (optional)
                - 'timestamp': [B*N,] tensor of timestamps (optional)
            viz_save_dir: Optional directory to save visualizations
        """
        pred_mean = metadata.get('pred_mean')
        target_mean = metadata.get('target_mean')
        gen_ensemble = metadata.get('gen_ensemble')
        target_ensemble = metadata.get('target_ensemble')  # For destandardized historical data
        timestamps = metadata.get('timestamp')
        hist_samples = metadata.get('hist_samples')  # [B*n, T_hist, N, F] STANDARDIZED
        
        if pred_mean is None or target_mean is None:
            logger.warning("⚠️  Cannot visualize: missing pred_mean or target_mean in metadata")
            return
        
        logger.info(f"📊 Starting visualization...")
        logger.info(f"   pred_mean shape: {pred_mean.shape}")
        logger.info(f"   pred_mean stats: mean={pred_mean.mean().item():.6f}, std={pred_mean.std().item():.6f}")
        logger.info(f"   target_mean stats: mean={target_mean.mean().item():.6f}, std={target_mean.std().item():.6f}")
        if hist_samples is not None:
            logger.info(f"   hist_samples shape: {hist_samples.shape}")
        
        # Ensure we have matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("⚠️  matplotlib not available, skipping visualization")
            return
        
        # Get stock symbols if available
        stock_symbols = self.dataset_info.get('stock_symbols', None) if self.dataset_info else None
        
        # Extract dimensions [B, T, N, F]
        B, T, N, F = pred_mean.shape
        
        # Visualize first few batches
        num_batches_to_viz = min(4, B)
        num_stocks_to_viz = min(5, N)
        
        # Randomly select stocks to visualize
        import random
        rng = random.Random(42)
        stock_indices = list(range(N))
        if N > num_stocks_to_viz:
            stock_indices = rng.sample(stock_indices, num_stocks_to_viz)
        else:
            stock_indices = stock_indices[:num_stocks_to_viz]
        
        # Get historical closing prices for visualization (not log returns!)
        hist_prices = None
        close_price = metadata.get('close_price')
        if close_price is not None:
            # close_price: [B_original*N, T_hist, 1] or [B_original, T_hist, N, 1]
            # where B_original = B*n (before ensemble split)
            if close_price.dim() == 3:  # [B_original*N, T_hist, 1]
                B_original = metadata.get('batch_size_original', metadata.get('batch_size', 1))
                N = metadata.get('num_stocks', 1)
                T_hist = close_price.shape[1]
                hist_prices = close_price.view(B_original, N, T_hist, 1).transpose(1, 2)  # [B_original, T_hist, N, 1]
                
                # Subsample to match actual batch size (ReplicatedDataset duplicates data n times)
                B = metadata.get('batch_size', 1)
                if B_original > B:
                    n = metadata.get('n_samples_per_input', 1)
                    hist_prices = hist_prices[::n, :, :, :]  # [B, T_hist, N, 1]
            elif close_price.dim() == 4:  # [B_original, T_hist, N, 1]
                hist_prices = close_price
                # Subsample if needed
                B = metadata.get('batch_size', 1)
                if hist_prices.shape[0] > B:
                    n = metadata.get('n_samples_per_input', 1)
                    hist_prices = hist_prices[::n, :, :, :]  # [B, T_hist, N, 1]
            logger.info(f"   Using historical closing prices: {hist_prices.shape if hist_prices is not None else None}")
        
        # Get price predictions for visualization
        pred_prices = metadata.get('pred_prices')  # [B, T, N, F]
        target_prices = metadata.get('target_prices')  # [B, T, N, F]
        gen_ensemble_prices = metadata.get('gen_ensemble_prices')  # [B, n, T, N, F] or None
        
        # Get n_samples_per_input to correctly index into raw (un-deduplicated) timestamps
        # timestamps shape: [B_original * N] = [B * n * N] where n = n_samples_per_input
        # After ensemble processing, batch_idx iterates over B unique timesteps,
        # so we must skip n*N entries per unique timestep (not just N).
        n_samples = metadata.get('n_samples_per_input', 1)
        if n_samples is None:
            n_samples = 1
        
        for batch_idx in range(num_batches_to_viz):
            # Extract timestamp for this batch if available
            time_start = None
            if timestamps is not None:
                # Each unique timestep occupies n*N consecutive entries in the raw timestamps tensor
                ts_idx = batch_idx * n_samples * N
                if ts_idx < timestamps.numel():
                    temp = timestamps[ts_idx]
                    time_start = temp.item() if hasattr(temp, 'item') else int(temp)
            
            # Get historical closing prices for this batch
            hist_batch = hist_prices[batch_idx] if hist_prices is not None else None  # [T_hist, N, 1]
            
            # Use prices if available, otherwise fall back to log returns
            if pred_prices is not None and target_prices is not None:
                pred_viz = pred_prices[batch_idx]  # [T, N, F]
                target_viz = target_prices[batch_idx]  # [T, N, F]
                gen_ens_viz = gen_ensemble_prices[batch_idx] if gen_ensemble_prices is not None else None  # [n, T, N, F]
            else:
                pred_viz = pred_mean[batch_idx]  # [T, N, F] log returns
                target_viz = target_mean[batch_idx]  # [T, N, F] log returns
                gen_ens_viz = gen_ensemble[batch_idx] if gen_ensemble is not None else None  # [n, T, N, F]
            
            self._plot_stock_predictions_v2(
                pred_mean=pred_viz,  # [T, N, F] - now PRICES
                target_mean=target_viz,  # [T, N, F] - now PRICES
                gen_ensemble=gen_ens_viz,  # [n, T, N, F] - now PRICES
                hist_samples=hist_batch,  # [T_hist, N, 1] - now PRICES
                stock_indices=stock_indices,
                stock_symbols=stock_symbols,
                batch_idx=batch_idx,
                time_start=time_start,
                viz_save_dir=viz_save_dir
            )
        
        # Paper-ready single-window forecast columns (Fig 1) — diffusion only,
        # written to a dedicated paper/ dir.  Existing diagnostics above are
        # unaffected.  No-op unless paper_figures.enabled and the active
        # baseline matches paper_figures.fig1.baseline.
        self._maybe_emit_paper_fig1(
            pred_prices=pred_prices,
            target_prices=target_prices,
            gen_ensemble_prices=gen_ensemble_prices,
            hist_prices=hist_prices,
            timestamps=timestamps,
            n_samples=n_samples,
            stock_symbols=stock_symbols,
            viz_save_dir=viz_save_dir,
        )

        logger.info(f"✅ Visualization complete")
    
    def _destandardize_historical_samples(self, hist_samples: torch.Tensor) -> torch.Tensor:
        """
        Destandardize historical samples using same stats as target samples.
        
        Args:
            hist_samples: [B, T_hist, N, F] standardized historical log returns
        
        Returns:
            Destandardized historical samples [B, T_hist, N, F]
        """
        if self.dataset_info is None:
            return hist_samples
        
        target_feature_idx = self.dataset_info.get('target_feature_idx', 0)
        feature_names = self.dataset_info.get('feature_names', [])
        per_stock_stats = self.dataset_info.get('per_stock_stats', {})
        
        if not feature_names or target_feature_idx >= len(feature_names):
            return hist_samples
        
        target_feature_name = feature_names[target_feature_idx]
        if target_feature_name not in per_stock_stats:
            return hist_samples
        
        stats = per_stock_stats[target_feature_name]
        mean_array = torch.from_numpy(stats['mean']).float().to(hist_samples.device)
        std_array = torch.from_numpy(stats['std']).float().to(hist_samples.device)
        
        # Broadcast for [B, T_hist, N, F]
        N = hist_samples.shape[2]
        mean_broadcasted = mean_array[:N].view(1, 1, N, 1)
        std_broadcasted = std_array[:N].view(1, 1, N, 1)
        
        return hist_samples * std_broadcasted + mean_broadcasted
    
    @staticmethod
    def _plot_stock_predictions_v2(
        pred_mean: torch.Tensor,
        target_mean: torch.Tensor,
        gen_ensemble: Optional[torch.Tensor],
        hist_samples: Optional[torch.Tensor],
        stock_indices: List[int],
        stock_symbols: Optional[List[str]],
        batch_idx: int,
        time_start: Optional[int],
        viz_save_dir: Optional[str]
    ) -> None:
        """
        Plot stock price predictions vs targets with historical context.
        
        Args:
            pred_mean: [T, N, F] predicted PRICES (not log returns)
            target_mean: [T, N, F] target PRICES
            gen_ensemble: [n, T, N, F] ensemble PRICES (optional)
            hist_samples: [T_hist, N, 1] historical closing PRICES (optional)
            stock_indices: List of stock indices to plot
            stock_symbols: List of all stock symbols (optional)
            batch_idx: Batch index for labeling
            time_start: Starting timestamp/date index for this batch (optional)
            viz_save_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        import os
        
        T, N, F = pred_mean.shape
        T_hist = hist_samples.shape[0] if hist_samples is not None else 0
        n_stocks = len(stock_indices)
        
        # Create consecutive date indices if time_start is available
        # time_start is the index of the FIRST historical observation, so:
        #   Historical dates: [time_start, ..., time_start + T_hist - 1]
        #   Forecast dates:   [time_start + T_hist, ..., time_start + T_hist + T - 1]
        date_labels = None
        if time_start is not None:
            hist_dates = list(range(time_start, time_start + T_hist))
            forecast_dates = list(range(time_start + T_hist, time_start + T_hist + T))
            date_labels = hist_dates + forecast_dates
        
        # Create figure with subplots for each stock
        fig, axes = plt.subplots(n_stocks, 1, figsize=(12, 3 * n_stocks), sharex=True)
        if n_stocks == 1:
            axes = [axes]
        
        for plot_idx, stock_idx in enumerate(stock_indices):
            ax = axes[plot_idx]
            
            # Extract price data for this stock: [T,], squeeze feature dim
            pred_stock = pred_mean[:, stock_idx, 0].cpu()  # [T,] - PRICES
            target_stock = target_mean[:, stock_idx, 0].cpu()  # [T,] - PRICES
            
            # Convert to numpy for plotting
            pred_prices_np = pred_stock.numpy()
            target_prices_np = target_stock.numpy()
            
            # Create time indices
            T_hist_actual = hist_samples.shape[0] if hist_samples is not None else 0
            T_forecast = len(pred_stock)
            
            # Use date labels if available, otherwise use integer indices
            if date_labels is not None:
                hist_time = date_labels[:T_hist_actual]
                forecast_time = date_labels[T_hist_actual:T_hist_actual + T_forecast]
            else:
                hist_time = list(range(T_hist_actual))
                forecast_time = list(range(T_hist_actual, T_hist_actual + T_forecast))
            
            # Historical data (if available)
            last_hist_price = None
            if hist_samples is not None:
                hist_stock = hist_samples[:, stock_idx, 0].cpu()  # [T_hist,] - PRICES
                hist_prices_np = hist_stock.numpy()
                
                # Plot historical closing prices
                ax.plot(hist_time, hist_prices_np, color='black', linewidth=1.5, label='Historical')
                last_hist_price = hist_prices_np[-1] if len(hist_prices_np) > 0 else None
            
            # Add connection line from last historical to first forecast if available
            if last_hist_price is not None and T_forecast > 0:
                # Draw dotted line connecting last history to first target/prediction
                connection_x = [hist_time[-1], forecast_time[0]]
                ax.plot(connection_x, [last_hist_price, target_prices_np[0]], 
                       ':', color='tab:blue', linewidth=1, alpha=0.5)
                ax.plot(connection_x, [last_hist_price, pred_prices_np[0]], 
                       ':', color='tab:orange', linewidth=1, alpha=0.5)
            
            # Plot ensemble predictions in background if available (before mean prediction)
            if gen_ensemble is not None:
                # gen_ensemble: [n, T, N, F] - PRICES
                ensemble_stock = gen_ensemble[:, :, stock_idx, 0].cpu()  # [n, T]
                n_ensemble = ensemble_stock.shape[0]
                
                # Plot each ensemble member with transparent orange, connecting to last historical price
                for i in range(n_ensemble):
                    ensemble_member = ensemble_stock[i].numpy()  # [T,]
                    
                    # Connect to last historical price if available for continuity
                    if last_hist_price is not None:
                        ensemble_x = [hist_time[-1]] + forecast_time
                        ensemble_y = [last_hist_price] + ensemble_member.tolist()
                    else:
                        ensemble_x = forecast_time
                        ensemble_y = ensemble_member
                    
                    ax.plot(ensemble_x, ensemble_y, '-', color='tab:orange', 
                           linewidth=0.8, alpha=0.25, zorder=1)
            
            # Plot predictions and targets (actual prices) - on top of ensemble
            ax.plot(forecast_time, target_prices_np, 'o-', color='tab:blue', linewidth=1.5, 
                   label='Target', markersize=5, alpha=0.8, zorder=3)
            ax.plot(forecast_time, pred_prices_np, 'x--', color='tab:orange', linewidth=1.5, 
                   label='Predicted (Mean)', markersize=6, alpha=0.8, zorder=4)
            
            # Add ensemble confidence bands if available
            if gen_ensemble is not None:
                # Compute 90% confidence interval on prices directly
                lower_band = torch.quantile(ensemble_stock, 0.05, dim=0).numpy()  # [T,]
                upper_band = torch.quantile(ensemble_stock, 0.95, dim=0).numpy()  # [T,]
                
                ax.fill_between(
                    forecast_time,
                    lower_band,
                    upper_band,
                    color='tab:orange',
                    alpha=0.2,
                    label='90% CI',
                    zorder=2
                )
            
            # Labels and title
            stock_label = f"{stock_symbols[stock_idx]}" if stock_symbols and stock_idx < len(stock_symbols) else f"Stock {stock_idx}"
            ax.set_ylabel('Price', fontsize=10)
            ax.set_title(f'{stock_label}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Format x-axis - show fewer ticks if there are many time steps
            if date_labels is not None and len(date_labels) > 15:
                # Show every Nth label
                n_ticks = 10
                step = max(1, len(date_labels) // n_ticks)
                tick_positions = [date_labels[i] for i in range(0, len(date_labels), step)]
                if plot_idx == len(stock_indices) - 1:  # Only set on last subplot
                    ax.set_xticks(tick_positions)
                    ax.tick_params(axis='x', rotation=45)
            
            # Legend only on first subplot
            if plot_idx == 0:
                ax.legend(loc='best', fontsize=9)
        
        axes[-1].set_xlabel('Date Index' if date_labels is not None else 'Time Step', fontsize=10)
        # plt.tight_layout()
        
        # Save figure
        if viz_save_dir is None:
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_cfg = HydraConfig.get()
                viz_save_dir = hydra_cfg.runtime.output_dir
            except Exception:
                viz_save_dir = None
        
        if viz_save_dir:
            os.makedirs(viz_save_dir, exist_ok=True)
            # Include timestamp in filename if available
            if time_start is not None:
                save_path = os.path.join(viz_save_dir, f"stock_prices_t{time_start}_b{batch_idx}.pdf")
            else:
                save_path = os.path.join(viz_save_dir, f"stock_prices_b{batch_idx}.pdf")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"   Saved visualization to: {save_path}")

            # Dump per-window plot-data sidecar for replot_baselines.
            try:
                from graph_signal_diffusion.evaluation.plot_data_io import (
                    dump_stock_prices_window,
                )
                _sidecar_dir = os.path.join(viz_save_dir, "stock_prices")
                if time_start is not None:
                    _sidecar_path = os.path.join(
                        _sidecar_dir, f"t{time_start}_b{batch_idx}.npz",
                    )
                else:
                    _sidecar_path = os.path.join(
                        _sidecar_dir, f"b{batch_idx}.npz",
                    )
                dump_stock_prices_window(
                    _sidecar_path,
                    pred_mean=pred_mean,
                    target_mean=target_mean,
                    gen_ensemble=gen_ensemble,
                    hist_samples=hist_samples,
                    stock_indices=stock_indices,
                    stock_symbols=stock_symbols,
                    batch_idx=batch_idx,
                    time_start=time_start,
                )
            except Exception:
                logger.warning(
                    "Failed to dump per-window stock_prices plot data sidecar"
                )

        plt.close(fig)

    # ------------------------------------------------------------------
    # Paper-ready figures (Fig 1: per-window forecast columns)
    # ------------------------------------------------------------------

    def _maybe_emit_paper_fig1(
        self,
        *,
        pred_prices: Optional[torch.Tensor],
        target_prices: Optional[torch.Tensor],
        gen_ensemble_prices: Optional[torch.Tensor],
        hist_prices: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
        n_samples: int,
        stock_symbols: Optional[List[str]],
        viz_save_dir: Optional[str],
    ) -> None:
        """Emit paper-ready single-window forecast columns (Fig 1).

        One single-column figure (``n_stocks`` rows sharing the x-axis) is
        written per test window, gated to the configured baseline (default
        ``diffusion``).  Up to ``max_panels`` figures are produced across the
        whole eval so a few can later be hand-picked and stacked side by side.
        Driven entirely by the ``paper_figures.fig1`` config block — a no-op
        when ``paper_figures.enabled`` is false/unset or the active baseline
        does not match.
        """
        import os
        import random

        pf = getattr(self, "paper_figures_cfg", None) or {}
        if not pf.get("enabled", False):
            return
        fig1_cfg = pf.get("fig1", {}) or {}
        if getattr(self, "_active_baseline_name", None) != fig1_cfg.get(
            "baseline", "diffusion"
        ):
            return
        if pred_prices is None or target_prices is None:
            return

        # Per-(baseline,split) emitted-panel counter, keyed by viz_save_dir.
        key = viz_save_dir or "_default_"
        counts = getattr(self, "_paper_fig1_counts", None)
        if counts is None:
            counts = {}
            self._paper_fig1_counts = counts
        emitted = counts.get(key, 0)

        max_panels = int(fig1_cfg.get("max_panels", 6))
        if emitted >= max_panels:
            return

        n_stocks = int(fig1_cfg.get("n_stocks", 3))
        vary = bool(fig1_cfg.get("vary_stocks_per_window", True))
        base_seed = int(fig1_cfg.get("seed", 42))

        B = int(pred_prices.shape[0])
        num_stocks_total = int(pred_prices.shape[2])
        n_pick = min(n_stocks, num_stocks_total)

        out_dir = getattr(self, "_paper_out_dir", None) or os.path.join(
            viz_save_dir or ".", "paper"
        )
        split_label = getattr(self, "_active_split", None) or "eval"

        for w in range(B):
            if emitted >= max_panels:
                break

            # Fresh random stock subset per emitted panel (for variety) unless
            # vary=false, in which case every window shows the same stocks.
            rng = random.Random(base_seed + (emitted if vary else 0))
            if num_stocks_total > n_pick:
                stock_idx = sorted(rng.sample(range(num_stocks_total), n_pick))
            else:
                stock_idx = list(range(num_stocks_total))

            # Window start timestamp (mirrors the diagnostic plot's indexing:
            # each unique window occupies n_samples*N consecutive raw entries).
            time_start = None
            if timestamps is not None:
                ts_idx = w * (n_samples or 1) * num_stocks_total
                if ts_idx < timestamps.numel():
                    temp = timestamps[ts_idx]
                    time_start = temp.item() if hasattr(temp, "item") else int(temp)

            try:
                column = self._select_window_column(
                    window_idx=w,
                    stock_idx=stock_idx,
                    pred_prices=pred_prices,
                    target_prices=target_prices,
                    gen_ensemble_prices=gen_ensemble_prices,
                    hist_prices=hist_prices,
                    stock_symbols=stock_symbols,
                    time_start=time_start,
                )
                fname = (
                    f"fig1_forecast_column_{split_label}_w{emitted:02d}"
                    + (f"_t{time_start}" if time_start is not None else "")
                    + ".pdf"
                )
                out_path = os.path.join(out_dir, fname)
                # Aesthetics come from plot_style.plots.stock_prices (so live and
                # sidecar regen match); behavioral knobs stay in fig1_cfg above.
                self._plot_paper_forecast_column(
                    column, out_path, getattr(self, "plot_style", None),
                )
                logger.info(f"   Saved paper Fig 1 panel to: {out_path}")
                # Tiny column sidecar (already-sliced n_stocks) so replot can
                # regenerate this exact panel without a live eval.
                try:
                    from graph_signal_diffusion.evaluation.plot_data_io import (
                        dump_paper_fig1_column,
                    )
                    dump_paper_fig1_column(
                        os.path.splitext(out_path)[0] + ".npz", column=column,
                    )
                except Exception:
                    logger.warning(
                        "Failed to dump paper Fig 1 column sidecar for window %s",
                        w,
                    )
            except Exception:
                logger.warning(
                    "Failed to emit paper Fig 1 panel for window %s", w
                )
            # Count the attempt either way so a single bad window cannot stall
            # the cap or repeatedly reuse the same stock seed.
            emitted += 1

        counts[key] = emitted

    @staticmethod
    def _select_window_column(
        *,
        window_idx: int,
        stock_idx: List[int],
        pred_prices: torch.Tensor,
        target_prices: torch.Tensor,
        gen_ensemble_prices: Optional[torch.Tensor],
        hist_prices: Optional[torch.Tensor],
        stock_symbols: Optional[List[str]],
        time_start: Optional[int],
    ) -> Dict[str, Any]:
        """Slice a single window + chosen stocks into a plain-numpy 'column'.

        Returns a dict consumed by :meth:`_plot_paper_forecast_column` with
        arrays shaped per *selected* stock:
          ``pred``/``target``: ``[T_f, n]``; ``ensemble``: ``[K, T_f, n]`` or
          ``None``; ``hist``: ``[T_hist, n]`` or ``None``.
        """
        def _np(t):
            return t.detach().cpu().numpy()

        w = window_idx
        idx = list(stock_idx)
        pred = _np(pred_prices[w][:, idx, 0])       # [T_f, n]
        target = _np(target_prices[w][:, idx, 0])   # [T_f, n]
        ensemble = (
            _np(gen_ensemble_prices[w][:, :, idx, 0])  # [K, T_f, n]
            if gen_ensemble_prices is not None else None
        )
        hist = (
            _np(hist_prices[w][:, idx, 0])  # [T_hist, n]
            if hist_prices is not None else None
        )

        symbols: List[str] = []
        for i in idx:
            if stock_symbols is not None and i < len(stock_symbols):
                symbols.append(str(stock_symbols[i]))
            else:
                symbols.append(f"Stock {i}")

        T_hist = hist.shape[0] if hist is not None else 0
        T_f = pred.shape[0]
        date_labels = None
        if time_start is not None:
            date_labels = list(range(time_start, time_start + T_hist + T_f))

        return {
            "stock_indices": idx,
            "stock_symbols": symbols,
            "pred": pred,
            "target": target,
            "ensemble": ensemble,
            "hist": hist,
            "time_start": time_start,
            "date_labels": date_labels,
        }

    @staticmethod
    def _plot_paper_forecast_column(
        column: Dict[str, Any],
        out_path: str,
        plot_style: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render one paper-ready single-window forecast figure.

        ``n`` stock panels are stacked in a single column sharing the x-axis;
        panels are sized to be roughly square (``panel_aspect``) at a target
        ``panel_width`` so several columns can later be tiled side by side.

        Self-styling: ``plot_style`` supplies the global look via
        ``matplotlib.rc_context`` and per-figure aesthetics via
        ``plot_style.plots.stock_prices`` (sizes, colors, fonts) — so the figure
        is identical whether drawn in a live run or regenerated from a sidecar.
        Returns the saved path.
        """
        import os
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from graph_signal_diffusion.evaluation.plot_style_apply import (
            resolve_paper_style,
        )

        rc, pstyle, _colors = resolve_paper_style(plot_style, "stock_prices")
        panel_w = float(pstyle.get("panel_width", 2.4))
        aspect = float(pstyle.get("panel_aspect", 1.0)) or 1.0
        dpi = int(pstyle.get("dpi", 300))
        _c = pstyle.get("colors") or {}
        col_hist = _c.get("historical", "black")
        col_target = _c.get("target", "tab:blue")
        col_pred = _c.get("predicted", "tab:orange")
        col_ens = _c.get("ensemble", "tab:orange")
        col_ci = _c.get("ci", "tab:orange")
        ci_alpha = float(pstyle.get("ci_alpha", 0.2))
        ens_alpha = float(pstyle.get("ensemble_alpha", 0.2))
        grid_alpha = float(pstyle.get("grid_alpha", 0.3))
        _f = pstyle.get("fonts") or {}
        fs_title = float(_f.get("title", 8)); fs_label = float(_f.get("label", 7))
        fs_legend = float(_f.get("legend", 5.5)); fs_tick = float(_f.get("tick", 6))

        pred = np.asarray(column["pred"])        # [T_f, n]
        target = np.asarray(column["target"])    # [T_f, n]
        ensemble = column.get("ensemble")
        ensemble = None if ensemble is None else np.asarray(ensemble)  # [K,T_f,n]
        hist = column.get("hist")
        hist = None if hist is None else np.asarray(hist)              # [T_hist,n]
        symbols = column.get("stock_symbols") or [
            f"Stock {i}" for i in range(pred.shape[1])
        ]
        date_labels = column.get("date_labels")

        n = pred.shape[1]
        T_f = pred.shape[0]
        T_hist = hist.shape[0] if hist is not None else 0

        if date_labels is not None and len(date_labels) >= T_hist + T_f:
            hist_time = list(date_labels[:T_hist])
            forecast_time = list(date_labels[T_hist:T_hist + T_f])
        else:
            hist_time = list(range(T_hist))
            forecast_time = list(range(T_hist, T_hist + T_f))

        panel_h = panel_w / aspect
        with matplotlib.rc_context(rc):
            fig, axes = plt.subplots(
                n, 1, figsize=(panel_w, max(1.0, n * panel_h)),
                sharex=True, squeeze=False,
            )
            axes = axes[:, 0]

            for r in range(n):
                ax = axes[r]
                tgt = target[:, r]
                prd = pred[:, r]

                last_hist = None
                if hist is not None:
                    h = hist[:, r]
                    ax.plot(hist_time, h, color=col_hist, linewidth=1.0,
                            label="Historical")
                    last_hist = h[-1] if len(h) else None

                # Dotted connectors from last observed price to first forecast.
                if last_hist is not None and T_f > 0:
                    ax.plot([hist_time[-1], forecast_time[0]], [last_hist, tgt[0]],
                            ":", color=col_target, linewidth=0.8, alpha=0.5)
                    ax.plot([hist_time[-1], forecast_time[0]], [last_hist, prd[0]],
                            ":", color=col_pred, linewidth=0.8, alpha=0.5)

                # Ensemble members (thin, behind mean/CI).
                if ensemble is not None:
                    ens = ensemble[:, :, r]  # [K, T_f]
                    for k in range(ens.shape[0]):
                        if last_hist is not None:
                            ax.plot([hist_time[-1]] + forecast_time,
                                    [last_hist] + ens[k].tolist(),
                                    "-", color=col_ens, linewidth=0.6,
                                    alpha=ens_alpha, zorder=1)
                        else:
                            ax.plot(forecast_time, ens[k], "-", color=col_ens,
                                    linewidth=0.6, alpha=ens_alpha, zorder=1)

                ax.plot(forecast_time, tgt, "o-", color=col_target, linewidth=1.2,
                        markersize=3, alpha=0.9, label="Target", zorder=3)
                ax.plot(forecast_time, prd, "x--", color=col_pred, linewidth=1.2,
                        markersize=3.5, alpha=0.9, label="Predicted (Mean)", zorder=4)

                if ensemble is not None:
                    lo = np.quantile(ensemble[:, :, r], 0.05, axis=0)
                    hi = np.quantile(ensemble[:, :, r], 0.95, axis=0)
                    ax.fill_between(forecast_time, lo, hi, color=col_ci,
                                    alpha=ci_alpha, label="90% CI", zorder=2)

                ax.set_title(str(symbols[r]), fontsize=fs_title, fontweight="bold")
                ax.set_ylabel("Price", fontsize=fs_label)
                ax.tick_params(labelsize=fs_tick)
                ax.grid(alpha=grid_alpha)
                if r == 0:
                    ax.legend(loc="best", fontsize=fs_legend, framealpha=0.85)

            # Thin shared x-axis ticks to ~6 labels.
            all_time = (hist_time + forecast_time) if hist is not None else forecast_time
            if len(all_time) > 8:
                step = max(1, len(all_time) // 6)
                axes[-1].set_xticks(all_time[::step])
            axes[-1].tick_params(axis="x", rotation=45)
            axes[-1].set_xlabel(
                "Date Index" if date_labels is not None else "Time Step",
                fontsize=fs_label,
            )

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        return out_path

    def evaluate_samples(
        self,
        generated_samples: torch.Tensor,
        real_samples: torch.Tensor,
        metadata: Dict[str, Any],
        viz_save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate samples with comprehensive SP500 destandardization (V2 - self-contained).
        
        Both generated and real samples are in STANDARDIZED space due to ReplicatedDataset.
        This method destandardizes both before computing three tiers of metrics:
          Tier 1 (primary): daily log returns — MAE, RMSE, MAPE, sMAPE, CRPS, direction acc
          Tier 2 (secondary): closing prices — MAE, RMSE, MSE, CRPS, per-horizon
          Tier 3 (cumulative): final-horizon cumulative log return — MAE, RMSE, CRPS, dir acc
        
        Args:
            generated_samples: Model outputs [B*n, T, N, F] from diffusion (STANDARDIZED)
            real_samples: Ground truth [B*n, T, N, F] from prepare_data_v2 (STANDARDIZED)
            metadata: Dict with stocks_index, batch_size, num_stocks, n_samples_per_input
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"="*60)
        logger.info(f"TaskV2 evaluate_samples: Starting evaluation pipeline")
        logger.info(f"="*60)
        
        # Step 1: Destandardize both samples using per-stock statistics
        #   → generated_destd, real_destd: [B*n, T, N, F]
        #   → also stores metadata['per_stock_mean']: [1, 1, N, 1]
        generated_destd, real_destd = self._destandardize_both_samples(
            generated_samples, real_samples, metadata
        )
        
        # Step 2: Reshape to ensemble format and compute diversity
        #   → metadata['gen_ensemble']:    [B, n, T, N, F]  (view, n at dim 1)
        #   → metadata['target_single']:   [B, T, N, F]     (first replica = true observation)
        #   → metadata['pred_mean']:       [B, T, N, F]     (ensemble mean of generated, mean over n dim=1)
        #   → metadata['target_mean']:     [B, T, N, F]     (for viz backward compat)
        diversity_metrics = self._process_ensemble_v2(generated_destd, real_destd, metadata)
        
        # Step 3: Single price conversion of full ensemble
        #   FIX #2: One conversion for the entire [B, n, T, N, F] ensemble instead of
        #   separate 4D + 5D calls. Derive mean prices in price space so that
        #   E[exp(X)] is correct (not exp(E[X]), which suffers from Jensen's inequality).
        gen_ensemble_lr = metadata['gen_ensemble']       # [B, n, T, N, F] log returns
        target_single_lr = metadata['target_single']     # [B, T, N, F] log returns
        
        # Convert full generated ensemble to prices (5D path, cumsum over dim=2=T)
        gen_ensemble_prices, _ = self._convert_log_returns_to_prices_v2(
            gen_ensemble_lr, gen_ensemble_lr, metadata
        )
        metadata['gen_ensemble_prices'] = gen_ensemble_prices  # [B, n, T, N, F]
        
        # Ensemble mean of PRICES (not exp of mean log returns)
        metadata['pred_prices'] = gen_ensemble_prices.mean(dim=1)  # [B, T, N, F] — mean over n (dim=1)
        
        # Convert single target to prices (4D path, cumsum over dim=1=T)
        _, target_prices = self._convert_log_returns_to_prices_v2(
            target_single_lr, target_single_lr, metadata
        )
        metadata['target_prices'] = target_prices  # [B, T, N, F]
        
        # Store log-return references for viz and tier-1 metrics
        metadata['pred_log_returns'] = metadata['pred_mean']            # [B, T, N, F]
        metadata['target_log_returns'] = metadata['target_single']      # [B, T, N, F]
        metadata['gen_ensemble_log_returns'] = metadata['gen_ensemble']  # [B, n, T, N, F]
        
        # Step 4: Compute three-tier metrics
        metrics = self._compute_three_tier_metrics(metadata)
        
        # Step 5: Add diversity metrics
        metrics.update(diversity_metrics)
        
        # Step 6: Visualize results
        self._visualize_results_v2(metadata, viz_save_dir)
        
        logger.info(f"="*60)
        logger.info(f"TaskV2 evaluate_samples: Complete")
        logger.info(f"="*60)
        
        return metrics
