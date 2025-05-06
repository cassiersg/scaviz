Scaviz
======

A side-channel visualization library based on
[fastplotlib](https://fastplotlib.org) and [pygfx](https://pygfx.org).

**Caution** This is havily work-in-progress, use at your own risk.

# Running examples

`scaviz` currently depends on forks of fastplotlib and pygfx. Therefore, to
install the correct version of dependencies, it is strongly recommended to use
[uv](https://docs.astral.sh/uv/).

## On desktop

```
uv run python examples/<example_name>.py
```

Alternatively, use `uv` to create a virtual environment:
```
uv run
source .venv/bin/activate
python examples/<example_name>.py
```

## Jupyter Lab

```
uv run --with jupyter --with jupyter_rfb --with simplejpeg jupyter lab
```

(then open the notebooks in `examples/`)
