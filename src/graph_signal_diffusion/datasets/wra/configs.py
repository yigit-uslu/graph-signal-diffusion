"""Configuration utilities for WRA datasets."""


def dataset_name_to_alias(dataset_name: str) -> str:
    """Convert a dataset identifier to a compact WRA alias for tagging/plotting.

    New-style dataset names are content-addressed paths like
    ``"debug/pdc_npz_abc123def456"`` (scenario component has ``wra_`` prefix stripped).
    The scenario component (before the first ``/``) is normalised to produce a stable alias:

    - ``"debug/pdc_npz_..."``              → ``"wra-debug"``
    - ``"small/pdc_npz_..."``              → ``"wra-small"``
    - ``"large_low_density/pdc_npz_..."``  → ``"wra-large-low-density"``
    - ``"wra-debug"``                      → ``"wra-debug"``  (already alias-like)
    - Unknown names                       → ``"wra-{name}"`` (graceful fallback)
    """
    if not isinstance(dataset_name, str) or len(dataset_name.strip()) == 0:
        return "wra-unknown"

    name = dataset_name.strip()
    base_name = name.split("/")[0]

    # Already alias-like.
    if base_name.startswith("wra-"):
        return base_name

    # Scenario component uses wra_ prefix — normalise to wra-.
    if base_name.startswith("wra_"):
        return base_name.replace("_", "-")

    # Fallback: normalise but keep some source signal.
    return f"wra-{base_name.replace('_', '-')}"
