"""Transition table for storing labelled transition data.

Provides the :class:`TransitionTable` dataclass, which stores transition
counts or probabilities as a labelled square matrix with convenient
access patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class _LocAccessor:
    """Key-based accessor for TransitionTable.

    Supports ``table.loc[from_key, to_key]`` lookups.
    """

    def __init__(self, table: TransitionTable) -> None:
        self._table = table

    def __getitem__(self, key: tuple[int | str, int | str]) -> int | float:
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
        value: int | float = self._table.matrix[i, j].item()
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
        matrix: A square 2-D numpy array of transition values.

    Raises:
        ValueError: If *matrix* is not 2-D and square, if
            ``len(keys) != matrix.shape[0]``, or if *keys* contains
            duplicates.
    """

    keys: tuple[int, ...] | tuple[str, ...] | tuple[()]
    matrix: np.ndarray
    _key_to_index: dict[int | str, int] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"matrix must be a square 2-D array, got shape {self.matrix.shape}"
            )
        if len(self.keys) != self.matrix.shape[0]:
            raise ValueError(
                f"len(keys) ({len(self.keys)}) != matrix dimension "
                f"({self.matrix.shape[0]})"
            )
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("keys must not contain duplicates")
        # frozen=True requires object.__setattr__ for init
        object.__setattr__(
            self, '_key_to_index',
            {k: i for i, k in enumerate(self.keys)},
        )
        self.matrix.flags.writeable = False

    @property
    def loc(self) -> _LocAccessor:
        """Key-based accessor.

        Usage::

            table.loc["A", "B"]  # value for A -> B
        """
        return _LocAccessor(self)

    def to_dict(self) -> dict[int | str, dict[int | str, int | float]]:
        """Convert to a square dict-of-dicts.

        Returns:
            A dict ``{from_key: {to_key: value}}`` mirroring the matrix.
        """
        return {
            self.keys[i]: {
                self.keys[j]: self.matrix[i, j].item()
                for j in range(len(self.keys))
            }
            for i in range(len(self.keys))
        }

    def reorder(self, keys: Sequence[int] | Sequence[str]) -> TransitionTable:
        """Return a new table with rows and columns reordered.

        Args:
            keys: The desired key ordering. Must contain exactly
                the same keys as the current table.

        Returns:
            A new :class:`TransitionTable` with reordered rows and columns.

        Raises:
            ValueError: If *keys* does not match the current key set exactly.
        """
        new_keys: tuple[int, ...] | tuple[str, ...] = tuple(keys)  # type: ignore[assignment]
        if set(new_keys) != set(self.keys):
            missing = set(self.keys) - set(new_keys)
            extra = set(new_keys) - set(self.keys)
            parts = []
            if missing:
                parts.append(f"missing keys: {missing!r}")
            if extra:
                parts.append(f"unknown keys: {extra!r}")
            raise ValueError(
                f"keys must be a permutation of the current keys; {'; '.join(parts)}"
            )
        order = [self._key_to_index[k] for k in new_keys]
        reordered = np.array(self.matrix)[np.ix_(order, order)]
        return TransitionTable(keys=new_keys, matrix=reordered)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransitionTable):
            return NotImplemented
        return self.keys == other.keys and np.array_equal(self.matrix, other.matrix)

    def __hash__(self) -> int:
        return hash((self.keys, self.matrix.tobytes()))

    def __repr__(self) -> str:
        return (
            f"TransitionTable(keys={self.keys!r}, "
            f"shape={self.matrix.shape[0]}x{self.matrix.shape[1]})"
        )
