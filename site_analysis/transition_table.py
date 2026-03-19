"""Transition table for storing labelled transition data.

Provides the :class:`TransitionTable` dataclass, which stores transition
counts or probabilities as a labelled square matrix with convenient
access patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class _LocAccessor:
    """Key-based accessor for TransitionTable.

    Supports ``table.loc[from_key, to_key]`` lookups.
    """

    def __init__(self, table: TransitionTable) -> None:
        self._table = table

    def __getitem__(self, key: tuple) -> int | float:
        from_key, to_key = key
        index = self._table._key_to_index
        try:
            i = index[from_key]
        except KeyError:
            raise KeyError(from_key) from None
        try:
            j = index[to_key]
        except KeyError:
            raise KeyError(to_key) from None
        value: int | float = self._table.data[i, j].item()
        return value


@dataclass(frozen=True)
class TransitionTable:
    """A labelled square matrix of transition data.

    Stores transition counts or probabilities with named keys for
    rows and columns. Provides multiple access patterns:

    - ``.matrix`` — the raw :class:`numpy.ndarray`
    - ``.loc[from_key, to_key]`` — key-based lookup
    - ``.to_dict()`` — square dict-of-dicts

    Args:
        keys: Row and column labels (site indices or site labels).
        data: A square 2-D numpy array of transition values.

    Raises:
        ValueError: If *data* is not 2-D and square, if
            ``len(keys) != data.shape[0]``, or if *keys* contains
            duplicates.
    """

    keys: list[int] | list[str]
    data: np.ndarray
    _key_to_index: dict[int | str, int] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError(
                f"data must be a square 2-D array, got shape {self.data.shape}"
            )
        if len(self.keys) != self.data.shape[0]:
            raise ValueError(
                f"len(keys) ({len(self.keys)}) != data dimension "
                f"({self.data.shape[0]})"
            )
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("keys must not contain duplicates")
        # frozen=True requires object.__setattr__ for init
        object.__setattr__(
            self, '_key_to_index',
            {k: i for i, k in enumerate(self.keys)},
        )

    @property
    def matrix(self) -> np.ndarray:
        """The transition data as a 2-D numpy array."""
        return self.data

    @property
    def loc(self) -> _LocAccessor:
        """Key-based accessor.

        Usage::

            table.loc["A", "B"]  # value for A -> B
        """
        return _LocAccessor(self)

    def to_dict(self) -> dict:
        """Convert to a square dict-of-dicts.

        Returns:
            A dict ``{from_key: {to_key: value}}`` mirroring the matrix.
        """
        return {
            self.keys[i]: {
                self.keys[j]: self.data[i, j].item()
                for j in range(len(self.keys))
            }
            for i in range(len(self.keys))
        }

    def __repr__(self) -> str:
        return (
            f"TransitionTable(keys={self.keys}, "
            f"shape={self.data.shape[0]}x{self.data.shape[1]})"
        )
