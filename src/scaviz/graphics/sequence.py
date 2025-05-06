import numpy as np

import pygfx

from fastplotlib.graphics._base import Graphic
from fastplotlib.graphics.selectors import LinearRegionSelector
from ..objects import Seq, SeqMaterial


def str2color(color: str | pygfx.Color):
    return pygfx.Color(color) if isinstance(color, str) else color


class SequenceGraphic(Graphic):
    def __init__(
        self,
        data: np.ndarray,
        thickness: float = 2.0,
        color: str | pygfx.Color = "w",
        size_space: str = "screen",
        x_scale: float = 1.0,
        z: float = 0.0,
        **kwargs,
    ):
        """
        Create a 2d sequence.

        Parameters
        ----------
        data: array-like
            Line sequence

        thickness: float, optional, default 2.0
            thickness of the line

        color: str or pygfx.Color, default "w"
            specify color as a single human-readable string or a pygfx Color object

        size_space: str, default "screen"
            coordinate space in which the thickness is expressed ("screen", "world", "model")

        x_scale: int, default 1.0
            x delta between consecutive points of the sequence

        z: float, default 0.0
            z coordinate of the plot

        **kwargs
            passed to Graphic
        """

        super().__init__(**kwargs)

        material = SeqMaterial(
            thickness=thickness,
            thickness_space=size_space,
            color=str2color(color),
            pick_write=True,
            aa=True,
        )

        world_object = Seq.from_sequence(
            data, z=z, f=8, material=material, x_scale=x_scale
        )

        self._set_world_object(world_object)

    @property
    def color(self) -> pygfx.Color:
        """Get or set the colors data"""
        return self.world_object.material.color

    @color.setter
    def color(self, value: str | pygfx.Color):
        self.world_object.material.color = str2color(value)

    def add_linear_region_selector(
        self,
        selection: tuple[float, float] | None = None,
        padding: float = 0.0,
        axis: str = "x",
        **kwargs,
    ) -> LinearRegionSelector:
        """
        Add a :class:`.LinearRegionSelector`. Selectors are just ``Graphic`` objects, so you can manage,
        remove, or delete them from a plot area just like any other ``Graphic``.

        Parameters
        ----------
        selection: (float, float), optional
            the starting bounds of the linear region selector, computed from data if not provided

        axis: str, default "x"
            axis that the selector resides on

        padding: float, default 0.0
            Extra padding to extend the linear region selector along the orthogonal axis to make it easier to interact with.

        kwargs
            passed to ``LinearRegionSelector``

        Returns
        -------
        LinearRegionSelector
            linear selection graphic

        """
        sz = self.world_object.size
        limits = 0, sz - 1

        dmin, dmax = self.world_object.minmax()
        size = int((dmax - dmin) + padding)
        center = (dmin + dmax) / 2

        if selection is None:
            selection = int(0.25 * sz), int(0.75 * sz)

        # create selector
        selector = LinearRegionSelector(
            selection=selection,
            limits=limits,
            size=size,
            center=center,
            axis=axis,
            parent=self,
            **kwargs,
        )
        self._plot_area.add_graphic(selector, center=False)
        # place selector below this graphic
        selector.offset = selector.offset + (0.0, 0.0, self.world_object.z - 1.0)
        return selector
