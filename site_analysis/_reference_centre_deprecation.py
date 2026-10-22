"""Deprecation warnings for sites without reference centres.

Polyhedral and dynamic Voronoi sites without a reference centre, and the
calls that create them, are deprecated and will be removed in version 2.0.
Each deprecated call warns once, at the user's line. The calls the package
makes on the user's behalf run inside ``on_users_behalf()``, so that the
sites and calls they create add no further warnings.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_on_users_behalf: ContextVar[bool] = ContextVar("_on_users_behalf", default=False)


def warn_without_reference_centres(message: str) -> None:
    """Issue a ``FutureWarning`` for a call without reference centres.

    Call this directly from the public function or method the user calls,
    so that the warning points at the user's call. Nothing is issued inside
    ``on_users_behalf()``.

    Args:
        message: The warning message.
    """
    if not _on_users_behalf.get():
        warnings.warn(message, FutureWarning, stacklevel=3)


@contextmanager
def on_users_behalf() -> Iterator[None]:
    """Silence ``warn_without_reference_centres`` within the block.

    The package runs the calls it makes on the user's behalf in this block.
    Those calls create sites without reference centres only when the user's
    own call does, and that call has already warned.

    Yields:
        None
    """
    token = _on_users_behalf.set(True)
    try:
        yield
    finally:
        _on_users_behalf.reset(token)
