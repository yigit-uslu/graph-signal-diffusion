"""Smoke test for GRW cumulative-NLL tensor exposure."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch_geometric.data import Batch, Data

from graph_signal_diffusion.baselines.stock_price_forecasting.grw import (
    GeometricRandomWalk,
)


def _make_batch(B: int, N: int, T: int, F: int = 1) -> Batch:
    graphs = []
    for _ in range(B):
        y = torch.randn(N, T, F)
        x = torch.randn(N, 6, 4)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        close_price = torch.rand(N, 6, 1) * 100 + 20
        close_price_y = torch.rand(N, T, 1) * 100 + 20
        stocks_index = torch.arange(N)
        timestamp = torch.zeros(N, dtype=torch.long)
        graphs.append(
            Data(
                x=x,
                y=y,
                edge_index=edge_index,
                close_price=close_price,
                close_price_y=close_price_y,
                stocks_index=stocks_index,
                timestamp=timestamp,
            )
        )
    return Batch.from_data_list(graphs)


def _make_loader(n_batches: int, B: int, N: int, T: int):
    return [_make_batch(B=B, N=N, T=T) for _ in range(n_batches)]


def test_grw_populates_cumulative_nll_without_viz_dir() -> None:
    """`last_nll_tensors['cum_nll_*']` must be available even without plotting."""
    torch.manual_seed(0)
    B, N, T = 2, 4, 4
    fit_loader = _make_loader(n_batches=3, B=B, N=N, T=T)
    eval_loader = _make_loader(n_batches=2, B=B, N=N, T=T)

    grw = GeometricRandomWalk(
        device="cpu",
        n_samples=2,
        evaluation={
            "nll_metrics": True,
            "structural_metrics": False,
            "log_every_n_batches": 0,
        },
    )
    grw.fit(fit_loader)

    mock_task = MagicMock()

    def _prepare_data(data):
        B_local = data.num_graphs
        N_local = data.y.size(0) // B_local
        T_local = data.y.size(1)
        F_local = data.y.size(2)
        samples = data.y.view(B_local, N_local, T_local, F_local).permute(0, 2, 1, 3)
        return {
            "samples": samples,
            "metadata": {
                "batch_size": B_local,
                "num_timesteps": T_local,
                "num_stocks": N_local,
                "num_features": F_local,
                "close_price": data.close_price,
                "close_price_y": data.close_price_y,
                "stocks_index": data.stocks_index,
                "timestamp": data.timestamp,
            },
        }

    mock_task.prepare_data.side_effect = _prepare_data
    mock_task.evaluate_samples.return_value = {"return_mae": 0.123}

    metrics = grw.evaluate(eval_loader, mock_task, viz_save_dir=None, eval_split_name="test")

    assert "return_mae" in metrics
    assert mock_task.evaluate_samples.call_count == len(eval_loader)
    assert hasattr(grw, "last_nll_tensors")
    assert grw.last_nll_tensors["cum_nll_real"] is not None
    assert grw.last_nll_tensors["cum_nll_gen"] is not None
    assert grw.last_nll_tensors["cum_nll_real"].ndim == 2
    assert grw.last_nll_tensors["cum_nll_gen"].ndim == 2
    assert grw.last_nll_tensors["cum_nll_real"].shape[1] == T
    assert grw.last_nll_tensors["cum_nll_gen"].shape[1] == T
