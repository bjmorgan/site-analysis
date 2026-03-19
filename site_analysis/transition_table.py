"""Transition table for storing labelled transition data.

Provides the :class:`TransitionTable` class, which stores transition
counts or probabilities as a labelled square matrix with convenient
access patterns.
"""

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

import numpy as np

TableKey = TypeVar('TableKey', int, str)


class TransitionTable(Generic[TableKey]):
    """A labelled square matrix of transition data.

    Stores transition counts or probabilities with named keys for
    rows and columns. Provides multiple access patterns:

    - ``.matrix`` — the raw (read-only) :class:`numpy.ndarray`
    - ``.get(from_key, to_key)`` — key-based lookup
    - ``.to_dict()`` — square dict-of-dicts
    - ``.reorder(keys)`` — return a new table with reordered rows/columns
    - ``.filter(keys)`` — return a new table with only the specified keys

    Args:
        keys: Row and column labels (site indices or site labels).
        matrix: A square 2-D numpy array of transition values.

    Raises:
        ValueError: If *matrix* is not 2-D and square, if
            ``len(keys) != matrix.shape[0]``, or if *keys* contains
            duplicates.
    """

    __slots__ = ('_keys', '_matrix', '_key_to_index', '_frozen')

    def __init__(
        self,
        keys: tuple[TableKey, ...],
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
        self._keys: tuple[TableKey, ...] = keys
        self._key_to_index: dict[TableKey, int] = {k: i for i, k in enumerate(keys)}
        self._matrix.flags.writeable = False
        self._frozen = True

    @property
    def keys(self) -> tuple[TableKey, ...]:
        """Row and column labels."""
        return self._keys

    @property
    def matrix(self) -> np.ndarray:
        """The transition data as a read-only 2-D numpy array."""
        return self._matrix

    def get(self, from_key: TableKey, to_key: TableKey) -> int | float:
        """Look up a single transition value by key.

        Args:
            from_key: The source key (row).
            to_key: The destination key (column).

        Returns:
            The transition value at ``(from_key, to_key)``.

        Raises:
            KeyError: If either key is not present in the table.
        """
        try:
            i = self._key_to_index[from_key]
        except KeyError:
            raise KeyError(from_key) from None
        try:
            j = self._key_to_index[to_key]
        except KeyError:
            raise KeyError(to_key) from None
        value: int | float = self._matrix[i, j].item()
        return value

    def to_dict(self) -> dict[TableKey, dict[TableKey, int | float]]:
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

    def reorder(self, keys: Sequence[TableKey]) -> TransitionTable[TableKey]:
        """Return a new table with rows and columns reordered.

        Args:
            keys: The desired key ordering. Must contain exactly
                the same keys as the current table.

        Returns:
            A new :class:`TransitionTable` with reordered rows and columns.

        Raises:
            ValueError: If *keys* does not match the current key set exactly.
        """
        new_keys: tuple[TableKey, ...] = tuple(keys)
        if len(new_keys) != len(self._keys) or set(new_keys) != set(self._keys):
            missing = [k for k in self._keys if k not in set(new_keys)]
            extra = [k for k in new_keys if k not in self._key_to_index]
            parts = []
            if missing:
                parts.append(f"missing keys: {missing!r}")
            if extra:
                parts.append(f"unknown keys: {extra!r}")
            raise ValueError(
                f"keys must be a permutation of the current keys; {'; '.join(parts)}"
            )
        order = [self._key_to_index[k] for k in new_keys]
        reordered = self._matrix[np.ix_(order, order)]
        return TransitionTable(keys=new_keys, matrix=reordered)

    def filter(self, keys: Sequence[TableKey]) -> TransitionTable[TableKey]:
        """Return a new table containing only the specified keys.

        Extracts the requested rows and columns without re-normalising.
        The order of *keys* is preserved in the result.

        Args:
            keys: The keys to retain. Must be a subset of the current
                keys with no duplicates.

        Returns:
            A new :class:`TransitionTable` with only the requested keys.

        Raises:
            ValueError: If *keys* contains unknown or duplicate keys.
        """
        new_keys: tuple[TableKey, ...] = tuple(keys)
        if len(new_keys) != len(set(new_keys)):
            raise ValueError("keys must not contain duplicates")
        unknown = [k for k in new_keys if k not in self._key_to_index]
        if unknown:
            raise ValueError(f"unknown keys: {unknown!r}")
        if len(new_keys) == 0:
            empty: np.ndarray = np.empty((0, 0), dtype=self._matrix.dtype)
            return TransitionTable(keys=new_keys, matrix=empty)
        order = [self._key_to_index[k] for k in new_keys]
        filtered = self._matrix[np.ix_(order, order)]
        return TransitionTable(keys=new_keys, matrix=filtered)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransitionTable):
            return NotImplemented
        return self._keys == other._keys and np.array_equal(self._matrix, other._matrix)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"TransitionTable(keys={self._keys!r}, "
            f"shape={self._matrix.shape[0]}x{self._matrix.shape[1]})"
        )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, '_frozen', False):
            raise AttributeError("TransitionTable is immutable")
        super().__setattr__(name, value)
