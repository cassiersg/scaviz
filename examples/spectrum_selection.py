"""
Plot of a trace with selection for computing the spectrum of a region.

The spectrum is in log-scale, with the x axis unit being the sampling frequency.

Pan plots with the mouse, scroll/shift+scroll for x/y zoom.
"""

import math
import numpy as np
import fastplotlib as fpl
import pygfx as gfx

from scaviz.graphics import SequenceGraphic
from scaviz.controllers import SplitXYController


class SpectrumPlot:
    def __init__(
        self, trace, subplot, trace_graphic, auto_scale=True, max_fft_size=5 * 10**5
    ):
        self._trace = trace
        self._subplot = subplot
        self._auto_scale = auto_scale
        self._spectrum_graphic = None
        self._max_fft_size = max_fft_size
        self._selector = trace_graphic.add_linear_region_selector()
        self._selector.add_event_handler("selection")(self._update)

    def _update(self, _):
        lb, ub = self._selector.selection
        lb_idx = max(0, math.floor(lb))
        ub_idx = min(len(self._trace), math.ceil(ub))
        size = ub_idx - lb_idx
        if 10 <= size <= self._max_fft_size:
            y_sel = self._trace[lb_idx:ub_idx]
            window = np.hamming(len(y_sel))
            spectrum = np.fft.rfft(y_sel * window)
            spectrum = np.abs(spectrum[: len(self._trace) // 2])
            spectrum[spectrum < 1] = 1.0
            spectrum = np.log10(spectrum)
            new_spectrum_graphic = SequenceGraphic(
                spectrum, color="red", x_scale=0.5 / len(self._trace)
            )
            if self._spectrum_graphic is not None:
                self._subplot.remove_graphic(self._spectrum_graphic)
            self._spectrum_graphic = new_spectrum_graphic
            self._subplot.add_graphic(new_spectrum_graphic)
            if self._auto_scale:
                self._subplot.auto_scale()
        elif self._spectrum_graphic is not None:
            self._spectrum_graphic.color = "grey"


# Fake trace
N = 10**6
x = np.linspace(0, N, N, dtype=np.float32)
a0 = N / 1000
np0 = 30
t0 = N / np0
t1 = 100
y = np.sin(2 * math.pi / t0 * x) * a0 + np.sin(2 * math.pi / t1 * x) * (a0 / 10)

# Figure setup with views/interaction control.
fig = fpl.Figure(shape=(2, 1), size=(700, 560))
for i in range(2):
    fig[i, 0].camera = gfx.OrthographicCamera(
        depth_range=(-1e3, 1e3), maintain_aspect=False
    )
    fig[i, 0].controller = SplitXYController()

# Setup trace sub-plot.
trace = SequenceGraphic(y, color="blue")
fig[0, 0].add_graphic(trace)

spectrum_plot = SpectrumPlot(y, fig[1, 0], trace)


fig.show()

if __name__ == "__main__":
    fpl.loop.run()
