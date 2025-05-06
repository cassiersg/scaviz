"""
Plot multiple traces.

Pan plot with the mouse, scroll/shift+scroll for x/y zoom.
"""

from math import pi
import numpy as np
import fastplotlib as fpl
import pygfx as gfx
import cmap

from scaviz.graphics import SequenceGraphic
from scaviz.controllers import SplitXYController

# Fake traces
N_TRACES = 3
N = 10**7
x = np.linspace(0, N, N, dtype=np.float32)
a0 = N / 1000
np0 = 30
t0 = N / np0
t1 = 100
phases = np.linspace(0, 1.1 * pi, N_TRACES)

print("Generating traces...", end="", flush=True)
noise = np.sin(2 * pi / t1 * x) * (a0 / 10)
ys = [np.sin(2 * pi / t0 * x - p) * a0 + noise for p in phases]
print("done.")

colormap = cmap.Colormap("seaborn:tab10")

fig = fpl.Figure(size=(700, 560))
fig[0, 0].camera = gfx.OrthographicCamera(
    depth_range=(-1e3, 1e3), maintain_aspect=False
)
fig[0, 0].controller = SplitXYController()

print("Generating graphics...", end="", flush=True)
seq_graphics = [
    fig[0, 0].add_graphic(SequenceGraphic(y, color=c))
    for y, c in zip(ys, colormap.color_stops.colors)
]
print("done.")


fig.show(maintain_aspect=False)

if __name__ == "__main__":
    fpl.loop.run()
