"""Individual transitions of atoms between sites.

This module provides ``TransitionEvents``, the transitions recorded in a
trajectory, one per change of an atom's site, as returned by
``Trajectory.transition_events()``, and ``TransitionEvent``, a single
transition.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any, NamedTuple, overload

import numpy as np


class TransitionEvent(NamedTuple):
    """One transition of an atom from one site to another.

    Attributes:
        frame: Index of the frame, from 0, in which the atom is first
            counted in its new site.
        timestep: Timestep label of that frame, or None if the trajectory
            does not have a label for every frame.
        entry_frame: Index of the frame, from 0, from which the atom stayed
            inside its new site, without leaving, up to ``frame``. Equal to
            ``frame`` without commitment.
        entry_timestep: Timestep label of the entry frame, or None if the
            trajectory does not have a label for every frame.
        atom: ``Atom.index`` of the atom.
        from_site: ``Site.index`` of the site the atom left.
        to_site: ``Site.index`` of the site the atom moved to.
    """

    frame: int
    timestep: int | None
    entry_frame: int
    entry_timestep: int | None
    atom: int
    from_site: int
    to_site: int


@dataclass(frozen=True, eq=False)
class TransitionEvents:
    """Transitions of atoms between sites, one element per transition.

    Each attribute is a read-only one-dimensional integer array, with one
    element per transition, except that ``timestep`` and ``entry_timestep``
    may be None. Indexing with an integer gives a ``TransitionEvent``;
    indexing with a slice, boolean mask or integer array gives a new
    ``TransitionEvents``, so that, for example,
    ``events[events.to_site == k]`` selects the transitions into site k.
    Equality (``==``) compares identity.

    Attributes:
        frame: Index of the frame, from 0, in which each atom is first
            counted in its new site.
        timestep: Timestep label of each event's frame, or None if the
            trajectory does not have a label for every frame.
        entry_frame: Index of the frame, from 0, from which each atom stayed
            inside its new site, without leaving, up to ``frame``. Equal to
            ``frame`` without commitment.
        entry_timestep: Timestep label of each entry frame, or None if the
            trajectory does not have a label for every frame.
        atom: ``Atom.index`` of the atom that moved.
        from_site: ``Site.index`` of the site the atom left.
        to_site: ``Site.index`` of the site the atom moved to.
    """

    frame: np.ndarray
    timestep: np.ndarray | None
    entry_frame: np.ndarray
    entry_timestep: np.ndarray | None
    atom: np.ndarray
    from_site: np.ndarray
    to_site: np.ndarray

    def __post_init__(self) -> None:
        """Store each field as a read-only one-dimensional int64 copy.

        Raises:
            ValueError: If a field is not one-dimensional, or the fields
                differ in length.
        """
        lengths: set[int] = set()
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None and field.name in ("timestep", "entry_timestep"):
                continue
            array = np.array(value, dtype=np.int64)
            if array.ndim != 1:
                raise ValueError(
                    f"{field.name} must be one-dimensional, not {array.ndim}-dimensional")
            array.flags.writeable = False
            object.__setattr__(self, field.name, array)
            lengths.add(len(array))
        if len(lengths) > 1:
            raise ValueError("All fields of TransitionEvents must have the same length")

    def __len__(self) -> int:
        """Number of transitions."""
        return len(self.frame)

    @overload
    def __getitem__(self, key: int | np.integer) -> TransitionEvent: ...

    @overload
    def __getitem__(self, key: slice | np.ndarray | list[Any]) -> TransitionEvents: ...

    def __getitem__(self, key: Any) -> TransitionEvent | TransitionEvents:
        """One transition for an integer key; the selected transitions otherwise."""
        if isinstance(key, (int, np.integer)):
            return TransitionEvent(
                frame=int(self.frame[key]),
                timestep=None if self.timestep is None else int(self.timestep[key]),
                entry_frame=int(self.entry_frame[key]),
                entry_timestep=(None if self.entry_timestep is None
                                else int(self.entry_timestep[key])),
                atom=int(self.atom[key]),
                from_site=int(self.from_site[key]),
                to_site=int(self.to_site[key]))
        return TransitionEvents(
            frame=self.frame[key],
            timestep=None if self.timestep is None else self.timestep[key],
            entry_frame=self.entry_frame[key],
            entry_timestep=None if self.entry_timestep is None else self.entry_timestep[key],
            atom=self.atom[key],
            from_site=self.from_site[key],
            to_site=self.to_site[key])

    def __iter__(self) -> Iterator[TransitionEvent]:
        """Iterate over the transitions as ``TransitionEvent`` records."""
        for i in range(len(self)):
            yield self[i]
