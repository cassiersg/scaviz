import numpy as np

import pygfx


class SplitXYController(pygfx.PanZoomController):
    """A controller to pan and zoom a camera in a 2D plane parallel to the screen.

    Controls:

    * Left mouse button: pan.
    * Right mouse button: zoom (if `camera.maintain_aspect==False`, zooms in both dimensions).
    * Fourth mouse button: quickzoom
    * wheel: zoom to point along X.
    * shift+wheel: zoom to point along Y.

    """

    _default_controls = pygfx.PanZoomController._default_controls | {
        "wheel": ("xy_zoom_to_point", "push", (-0.001, 0.0)),
        "shift+wheel": ("xy_zoom_to_point", "push", (0.0, -0.001)),
    }

    def xy_zoom_to_point(
        self, delta: tuple[float, float], pos: tuple, rect: tuple, *, animate=False
    ):
        """Zoom the view while panning to keep the position under the cursor fixed.

        If animate is True, the motion is damped. This requires the
        controller to receive events from the renderer/viewport.
        """

        if animate:
            action_tuple = ("zoom_to_point", "push", 1.0)
            action = self._create_action(None, action_tuple, (0.0, 0.0), None, rect)
            action.set_target(delta)
            action.done = True
        elif self._cameras:
            self._update_zoom_to_point(delta, screen_pos=pos, rect=rect)
            return self._update_cameras()

    def _update_xy_zoom_to_point(self, delta, *, screen_pos, rect):
        dx, dy = delta

        fx, fy = 2**dx, 2**dy

        new_cam_state = self._zoom(fx, fy, self._get_camera_state())
        self._set_camera_state(new_cam_state)

        pan_delta = self._get_panning_to_compensate_xy_zoom(fx, fy, screen_pos, rect)
        vecx, vecy = self._get_camera_vecs(rect)
        self._update_pan(pan_delta, vecx=vecx, vecy=vecy)

    def _get_panning_to_compensate_xy_zoom(self, mulx, muly, screen_pos, rect):
        multiplier = np.array([mulx, muly])
        # Get viewport info
        x, y, w, h = rect

        # Distance from the center of the rect
        delta_screen_x = screen_pos[0] - x - w / 2
        delta_screen_y = screen_pos[1] - y - h / 2
        delta_screen1 = np.array([delta_screen_x, delta_screen_y])

        # New position after zooming
        delta_screen2 = delta_screen1 * multiplier

        # The amount to pan is the difference, but also scaled with the multiplier
        # because pixels take more/less space now.
        return tuple((delta_screen1 - delta_screen2) / multiplier)
