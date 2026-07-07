"""Tests for RevIN flag cross-validation at trainer startup."""

import logging
import pytest
from omegaconf import OmegaConf

from graph_signal_diffusion.cli.train import _validate_revin_flags


def _cfg(revin=False, space_align=False, has_flag=True):
    dataset = {"standardize_target_in_x_for_revin": space_align} if has_flag else {}
    return OmegaConf.create({
        "diffusion": {"revin": revin},
        "dataset": dataset,
    })


class TestRevinFlagCrosscheck:

    def test_both_off_ok(self):
        _validate_revin_flags(_cfg(False, False), logging.getLogger())

    def test_both_on_ok(self):
        _validate_revin_flags(_cfg(True, True), logging.getLogger())

    def test_revin_without_space_align_raises(self):
        with pytest.raises(ValueError, match="requires dataset.standardize_target_in_x"):
            _validate_revin_flags(_cfg(True, False), logging.getLogger())

    def test_space_align_without_revin_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _validate_revin_flags(_cfg(False, True), logging.getLogger("test"))
        assert "has no effect" in caplog.text

    def test_revin_on_dataset_without_flag_ok(self):
        """Datasets that don't define the flag (e.g. WRA) are unconstrained."""
        _validate_revin_flags(_cfg(revin=True, has_flag=False), logging.getLogger())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
