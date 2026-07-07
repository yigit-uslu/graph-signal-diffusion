import signal

import pytest

from graph_signal_diffusion.cli.train import (
    _is_graceful_system_exit,
    _map_sigterm_to_keyboard_interrupt,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (None, True),
        (0, True),
        (130, True),
        (143, True),
        ("130", True),
        ("143", True),
        (1, False),
        ("1", False),
        ("error", False),
    ],
)
def test_is_graceful_system_exit(code, expected):
    assert _is_graceful_system_exit(SystemExit(code)) is expected


def test_map_sigterm_to_keyboard_interrupt_restores_handler():
    previous_handler = signal.getsignal(signal.SIGTERM)
    with _map_sigterm_to_keyboard_interrupt():
        current_handler = signal.getsignal(signal.SIGTERM)
        assert current_handler != previous_handler
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            current_handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) == previous_handler
