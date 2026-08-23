"""Teacher qualification dataset and synthetic-data tooling."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataset import AmarkenDataset, DatasetSplit, PackedConversationDataset

__all__ = ("AmarkenDataset", "DatasetSplit", "PackedConversationDataset")


def __getattr__(name: str) -> Any:
    """Load dataset exports lazily so ``python -m src.data.dataset`` is clean."""
    if name not in __all__:
        raise AttributeError(name)
    from . import dataset

    return getattr(dataset, name)
