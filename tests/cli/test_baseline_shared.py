"""Unit tests for shared baseline CLI helpers."""
from graph_signal_diffusion.cli._baseline_shared import to_trainer_key


def test_to_trainer_key():
    assert to_trainer_key("val", "crps_mean") == "val_crps_mean"
    assert to_trainer_key("test", "price_rmse") == "test_price_rmse"
    assert to_trainer_key("train-val", "price_mae") == "train-val_price_mae"
