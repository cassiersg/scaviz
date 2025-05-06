"""Time sequence shader. Based on pygfx line shader."""

import math
import importlib.resources

import wgpu  # only for flags/enums
import numpy as np
import numpy.typing as npt
import pylinalg as la

from pygfx import Geometry
from pygfx.utils import array_from_shadertype
from pygfx.resources import Buffer
from pygfx.objects import WorldObject
from pygfx.materials import Material
from pygfx.utils import unpack_bitfield, Color
from pygfx.utils.enums import CoordSpace

from pygfx.renderers.wgpu import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    RenderMask,
)


# TODO: improve shader to avoid zig-zag line, then take line thickness into account


def _rec_pooling(x, f, func):
    poolings = [x]
    while len(poolings[-1]) >= f:
        x = poolings[-1]
        end_len = len(x) % f
        x_main = x[: len(x) - end_len]
        x_rem = x[len(x) - end_len :]
        x_res = x_main.reshape((len(x_main) // f, f))
        x_pooled = func(x_res, axis=1)
        if len(x_rem) != 0:
            x_rem_pooled = func(x_rem)
            x_pooled = np.hstack([x_pooled, x_rem_pooled])
        poolings.append(x_pooled)
    return poolings[1:]


class Seq(WorldObject):
    """Geometry.seq, self.z"""

    def __init__(self, *, xstep, xoffset, psizes, poffsets, z=0.0, **kwargs):
        super().__init__(**kwargs)
        self.z = z
        self.xstep = xstep
        self.xoffset = xoffset
        self.psizes = psizes
        self.poffsets = poffsets

    @property
    def size(self):
        return self.psizes[0]

    def minmax(self):
        offset = self.poffsets[-1]
        sz = self.psizes[-1]
        data = self.geometry.pooled_seq.data[offset : offset + sz]
        return np.min(data), np.max(data)

    def get_bounding_box(self):
        dmin, dmax = self.minmax()
        return np.array(
            [
                [self.xoffset[0], dmin, self.z],
                [self.xoffset[0] + self.xstep[0] * (self.size - 1), dmax, self.z],
            ]
        )

    @classmethod
    def from_sequence(
        cls, x: npt.ArrayLike, *, z=0.0, x_scale=1.0, f: int = 8, material, **kwargs
    ):
        assert isinstance(f, int)
        pooled_seqs = [
            np.vstack([xmin, xmax]).T.ravel()
            for xmin, xmax in zip(
                _rec_pooling(x, f, np.min), _rec_pooling(x, f, np.max)
            )
        ]
        px_sizes = np.array([len(x)] + [len(px) for px in pooled_seqs])
        px_offsets = np.cumsum([0] + list(px_sizes)[:-1])
        px_xsteps = [1.0] + [float(f ** (i + 1)) / 2 for i in range(len(pooled_seqs))]
        px_xsteps = list(x_scale * np.array(px_xsteps))
        px_xoffsets = [0.0] + [step / 2 for step in px_xsteps[1:]]
        pooled_seq = np.hstack([x] + pooled_seqs)
        assert len(px_sizes) == len(px_offsets)
        assert len(px_sizes) == len(px_xsteps)
        assert len(px_sizes) == len(px_xoffsets)

        geometry = Geometry(pooled_seq=pooled_seq.astype(np.float32))

        return cls(
            xstep=px_xsteps,
            xoffset=px_xoffsets,
            psizes=px_sizes,
            poffsets=px_offsets,
            geometry=geometry,
            material=material,
            z=z,
            **kwargs,
        )


class SeqMaterial(Material):
    """Basic line material.

    Parameters
    ----------
    thickness : float
        The line thickness expressed in logical pixels. Default 2.0.
    thickness_space : str | CoordSpace
        The coordinate space in which the thickness is expressed ('screen', 'world', 'model'). Default 'screen'.
    color : Color
        The uniform color of the line.
    aa : bool
        Whether or not the line is anti-aliased in the shader. Default True.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class <pygfx.Material>`.
    """

    uniform_type = dict(
        Material.uniform_type,
        color="4xf4",
        thickness="f4",
        dash_offset="f4",
    )

    def __init__(
        self,
        thickness=2.0,
        thickness_space="screen",
        *,
        color=(1, 1, 1, 1),
        aa=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.thickness = thickness
        self.thickness_space = thickness_space
        self.color = color
        self.aa = aa

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        # TODO change this
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, coord=18)
        return {
            "vertex_index": values["index"],
            "segment_coord": (values["coord"] - 100000) / 100000.0,
        }

    @property
    def color(self):
        """The uniform color of the line."""
        return Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_full()
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        return self._store.color_is_transparent

    @property
    def aa(self):
        return self._store.aa

    @aa.setter
    def aa(self, aa):
        self._store.aa = bool(aa)

    @property
    def thickness(self):
        """The line thickness.

        The interpretation depends on `thickness_space`. By default it is in logical
        pixels, but it can also be in world or model coordinates.
        """
        return float(self.uniform_buffer.data["thickness"])

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = max(0.0, float(thickness))
        self.uniform_buffer.update_full()

    @property
    def thickness_space(self):
        """The coordinate space in which the thickness (and dash_pattern) are expressed.

        See :obj:`pygfx.utils.enums.CoordSpace`:
        """
        return self._store.thickness_space

    @thickness_space.setter
    def thickness_space(self, value):
        value = value or "screen"
        if value not in CoordSpace:
            raise ValueError(
                f"LineMaterial.thickness_space must be a string in {CoordSpace}, not {value!r}"
            )
        self._store.thickness_space = value


@register_wgpu_render_function(Seq, SeqMaterial)
class SeqShader(BaseShader):
    type = "render"

    renderer_uniform_type = dict(
        seq_offset="i4",
        last_i="i4",
        z="f4",
        xstep="f4",
        xoffset="f4",
    )

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material

        self["line_type"] = "line"
        self["thickness_space"] = material.thickness_space
        self["aa"] = material.aa

        # FIXME: cleanup shader
        self["instanced"] = False
        self["loop"] = False
        self["dashing"] = False
        self["debug"] = False

        # Handle color
        self["color_mode"] = "uniform"

        self["color_buffer_channels"] = 0

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.renderer_uniform_type), force_contiguous=True
        )

    def get_bindings(self, wobject, shared):
        material = wobject.material

        pooled_seq = wobject.geometry.pooled_seq

        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("u_renderer", "buffer/uniform", self.uniform_buffer),
            Binding("s_pooled_seq", rbuffer, pooled_seq, "VERTEX"),
        ]

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        # Instanced lines have an extra storage buffer that we add manually
        bindings1 = {}  # non-auto-generated bindings

        return {
            0: bindings,
            1: bindings1,
        }

    def get_pipeline_info(self, wobject, shared):
        # Cull backfaces so that overlapping faces are not drawn.
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def _get_scale(self, wobject, shared, world_sizex):
        stdinfo_data = shared.uniform_data
        screen_sizex = stdinfo_data["logical_size"][0]
        i_sel = 0
        for i, xstep in enumerate(wobject.xstep):
            if screen_sizex <= world_sizex / xstep / 2.0:
                i_sel = i
            else:
                break
        return i_sel

    def _update_uniform_buffer(self, wobject, i_sel):
        new_data = self.uniform_buffer.data.copy()
        new_data["seq_offset"] = wobject.poffsets[i_sel]
        new_data["last_i"] = wobject.psizes[i_sel]
        new_data["z"] = wobject.z
        new_data["xstep"] = wobject.xstep[i_sel]
        if i_sel is None:
            self.uniform_buffer.data["xoffset"] = 0.0
        else:
            self.uniform_buffer.data["xoffset"] = wobject.xstep[i_sel] / 2
        if new_data.tobytes() != self.uniform_buffer.data.tobytes():
            self.uniform_buffer.set_data(new_data)

    def _world_rangex(self, shared):
        stdinfo_data = shared.uniform_data
        ndc_to_world = la.mat_inverse(
            stdinfo_data["cam_transform"] @ stdinfo_data["projection_transform"]
        )
        x = (-1.0, 0.0, 0.0, 1.0) @ ndc_to_world
        world_xmin = x[0] / x[3]
        x = (1.0, 0.0, 0.0, 1.0) @ ndc_to_world
        world_xmax = x[0] / x[3]
        return world_xmin, world_xmax

    def _render_range(self, wobject, world_xmin, world_xmax, i_sel):
        seq_len = int(wobject.psizes[i_sel])
        xstep = wobject.xstep[i_sel]
        xoffset = wobject.xoffset[i_sel]
        range_start = max(0, math.floor((world_xmin - xoffset) / xstep))
        range_end = math.ceil((world_xmax - xoffset) / xstep)
        offset = range_start
        size = max(0, min(seq_len - offset, range_end - range_start + 1))
        assert isinstance(offset, int)
        assert isinstance(size, int), f"{size=}"
        return offset, size

    def get_render_info(self, wobject, shared):
        material = wobject.material

        world_xmin, world_xmax = self._world_rangex(shared)
        i_sel = self._get_scale(wobject, shared, world_xmax - world_xmin)
        line_offset, line_size = self._render_range(
            wobject, world_xmin, world_xmax, i_sel
        )
        offset, size = 6 * line_offset, 6 * line_size

        self._update_uniform_buffer(wobject, i_sel)

        n_instances = 1

        render_mask = 0
        if wobject.render_mask:
            render_mask = wobject.render_mask
        elif material.is_transparent:
            render_mask = RenderMask.transparent
        else:
            # Get what passes are needed for the color
            if material.color_is_transparent:
                render_mask |= RenderMask.transparent
            else:
                render_mask |= RenderMask.opaque
            # Need transparency for aa
            if material.aa:
                render_mask |= RenderMask.transparent

        # print(f"pooled_seq get_render_info {world_xmax-world_xmin=}")
        # import traceback

        # traceback.print_stack()

        return {
            "indices": (size, n_instances, offset, 0),
            "render_mask": render_mask,
        }

    def get_code(self):
        return importlib.resources.read_text(__package__, "seq.wgsl")
