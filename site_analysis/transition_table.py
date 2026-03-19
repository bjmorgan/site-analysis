"""Transition table for storing labelled transition data.

Provides the :class:`TransitionTable` class, which stores transition
counts or probabilities as a labelled square matrix with convenient
access patterns.
"""

from __future__ import annotations

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


class TransitionTable:
    """A labelled square matrix of transition data.

    Stores transition counts or probabilities with named keys for
    rows and columns. Provides multiple access patterns:

    - ``.matrix`` — the raw (read-only) :class:`numpy.ndarray`
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

    __slots__ = ('_keys', '_matrix', '_key_to_index', '_loc')

    def __init__(
        self,
        keys: tuple[int, ...] | tuple[str, ...] | tuple[()],
        matrix: np.ndarray,
    ) -> None:
        self._matrix = np.array(matrix, copy=True)
        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"matrix must be a square 2-D array, got shape {self._matrix.shape}"
            )
        if len(keys) != self._matrix.shape[0]:
            raise ValueError(
                f"len(keys) ({len(keys)}) != matrix dimension "
                f"({self._matrix.shape[0]})"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("keys must not contain duplicates")
        self._keys = keys
        self._key_to_index = {k: i for i, k in enumerate(keys)}
        self._loc = _LocAccessor(self)
        self._matrix.flags.writeable = False

    @property
    def keys(self) -> tuple[int, ...] | tuple[str, ...] | tuple[()]:
        """Row and column labels."""
        return self._keys

    @property
    def matrix(self) -> np.ndarray:
        """The transition data as a read-only 2-D numpy array."""
        return self._matrix

    @property
    def loc(self) -> _LocAccessor:
        """Key-based accessor.

        Usage::

            table.loc["A", "B"]  # value for A -> B
        """
        return self._loc

    def to_dict(self) -> dict[int | str, dict[int | str, int | float]]:
        """Convert to a square dict-of-dicts.

        Returns:
            A dict ``{from_key: {to_key: value}}`` mirroring the matrix.
        """
        return {
            self._keys[i]: {
                self._keys[j]: self._matrix[i, j].item()
                for j in range(len(self._keys))
            }
            for i in range(len(self._keys))
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
        if set(new_keys) != set(self._keys):
            missing = set(self._keys) - set(new_keys)
            extra = set(new_keys) - set(self._keys)
            parts = []
            if missing:
                parts.append(f"missing keys: {missing!r}")
            if extra:
                parts.append(f"unknown keys: {extra!r}")
            raise ValueError(
                f"keys must be a permutation of the current keys; {'; '.join(parts)}"
            )
        order = [self._key_to_index[k] for k in new_keys]
        reordered = np.array(self._matrix)[np.ix_(order, order)]
        return TransitionTable(keys=new_keys, matrix=reordered)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransitionTable):
            return NotImplemented
        return self._keys == other._keys and np.array_equal(self._matrix, other._matrix)

    def __hash__(self) -> int:
        return hash((self._keys, self._matrix.tobytes()))

    def __repr__(self) -> str:
        return (
            f"TransitionTable(keys={self._keys!r}, "
            f"shape={self._matrix.shape[0]}x{self._matrix.shape[1]})"
        )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, '_loc'):
            raise AttributeError("TransitionTable is immutable")
        super().__setattr__(name, value)
