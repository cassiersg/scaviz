import pygfx


class PlotCamera(pygfx.OrthographicCamera):
    def __init__(
        self,
        z=1e3,
        width=1,
        height=1,
        *,
        zoom=1,
        maintain_aspect=True,
        depth_range=None,
    ):
        """Camera with fixed z position (`z`).

        See `pygfx.OrthographicCamera for other arguments.
        """
        super().__init__(
            width,
            height,
            zoom=zoom,
            maintain_aspect=maintain_aspect,
            depth_range=depth_range,
        )
        self._fixed_z = z
        self.local.z = z

    def set_state(self, state):
        if "position" in state:
            for i, k in enumerate(("x", "y", "z")):
                state[k] = state["position"][i]
        state = {k: v for k, v in state.items() if k != "position"}
        state["z"] = self._fixed_z
        print(f"PlotCamera.set_state, {state=}")
        res = super().set_state(state)
        print(f"{self.get_state()=}")
        return res
