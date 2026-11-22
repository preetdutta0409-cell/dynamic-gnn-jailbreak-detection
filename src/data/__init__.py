"""Dataset loading, validation, and scenario-grouped splitting utilities."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Dataset": ("dataset_loader", "Dataset"),
    "DatasetLoadError": ("dataset_loader", "DatasetLoadError"),
    "load_dataset": ("dataset_loader", "load_dataset"),
    "DatasetValidationError": ("dataset_validator", "DatasetValidationError"),
    "trainable_action_ids": ("dataset_validator", "trainable_action_ids"),
    "validate_dataset": ("dataset_validator", "validate_dataset"),
    "split_by_scenario": ("split_dataset", "split_by_scenario"),
    "write_split_manifest": ("split_dataset", "write_split_manifest"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public helpers lazily so module CLIs run without import warnings."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
