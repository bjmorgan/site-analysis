"""Plotting utilities for site-analysis tutorials."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    ax: plt.Axes | None = None,
    **kwargs,
) -> mpl.image.AxesImage:
    """Create a heatmap from a numpy array and two lists of labels.

    Args:
        data: 2-D array of values to plot.
        row_labels: Labels for the rows (y-axis).
        col_labels: Labels for the columns (x-axis).
        ax: Axes to plot on. Uses current axes if ``None``.
        **kwargs: Passed to :func:`matplotlib.axes.Axes.imshow`.

    Returns:
        The :class:`~matplotlib.image.AxesImage` instance.
    """
    if ax is None:
        ax = plt.gca()

    masked = np.ma.masked_equal(data, 0)
    cmap = plt.get_cmap(kwargs.pop("cmap", None))
    cmap.set_bad(color="0.85")
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, **kwargs)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Grid lines between cells.
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(
    im: mpl.image.AxesImage,
    data: np.ndarray | None = None,
    valfmt: str = "{x:.2f}",
    textcolours: tuple[str, str] = ("black", "white"),
    threshold: float | None = None,
    **textkw,
) -> list[mpl.text.Text]:
    """Annotate each cell of a heatmap with a formatted value.

    Args:
        im: The :class:`~matplotlib.image.AxesImage` returned by
            :func:`heatmap`.
        data: Data array. If ``None``, taken from *im*.
        valfmt: Format string using ``{x}`` as the placeholder.
        textcolours: Pair of colours ``(below_threshold, above_threshold)``.
        threshold: Value above which *textcolours[1]* is used. Defaults
            to the midpoint of the colour map range.
        **textkw: Passed to :func:`matplotlib.axes.Axes.text`.

    Returns:
        A list of :class:`~matplotlib.text.Text` instances.
    """
    if data is None:
        data = im.get_array()

    if threshold is None:
        threshold = im.norm(data.max()) / 2.0

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    formatter = mpl.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0:
                colour = textcolours[int(im.norm(data[i, j]) > threshold)]
                text = im.axes.text(
                    j, i, formatter(data[i, j], None), color=colour, **kw
                )
                texts.append(text)
    return texts
