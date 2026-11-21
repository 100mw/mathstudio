"""Lorentz transformation diagram for VS Code and Colab notebooks.

Coordinates use c = 1: the vertical coordinate is ct, and v means velocity/c.
"""

from math import nan

from mathstudio import ListPlot, ParametricPlot, Plot, Slider, sqrt, u

__all__ = ['SpacetimeDiagram']


def _segments(segments, **style):
    # NaN gaps let disconnected lines share one SVG trace.
    points = [point for start, end in segments for point in (start, end, (nan, nan))]
    x, y = zip(*points)
    ListPlot(x, y, points=0, lines=True, **style)


def SpacetimeDiagram(*, length=0, time=0, v=0, grid=True,
                     width=700, height=700, show=True):
    """Display the lab diagram with live length, time, and velocity sliders.

    length and time are the event's primed coordinates; v is the moving
    frame's velocity in units of c. Inputs set initial slider values.
    The returned Plot exposes .figure, .sliders, and .at(length=..., v=...).
    """
    length = Slider('length', 0, 6, 0.125, length)
    time = Slider('time', 0, 6, 0.125, time)
    v = Slider('v', 0, 0.99, 0.01, v)

    # Map primed coordinates to the home frame.
    gamma = 1 / sqrt(1 - v**2)

    def boost(x, ct):
        return gamma * (x + v*ct), gamma * (v*x + ct)

    lower, upper = -0.5, 6.25
    ticks = range(1, 7)

    def guide_start(fixed):
        return lower/gamma - v*fixed

    with Plot(x=(lower, upper), y=(lower, upper), width=width, height=height,
              xlabel='', ylabel='', show=show) as diagram:
        for axis in ('xaxis', 'yaxis'):
            diagram.layout[axis].update(
                zerolinecolor='gray', zerolinewidth=1.5, tick0=0, dtick=1,
                minor=dict(tick0=0, dtick=0.25, showgrid=grid,
                           gridcolor='#181818', gridwidth=0.5, ticks=''),
            )
        if grid:
            # Sample the visible branches directly from their equations.
            for i in ticks:
                bound = sqrt(upper**2 - i**2)
                branch = sqrt(u**2 + i**2)
                ParametricPlot(branch, u, interval=(lower, bound), samples=120,
                               color='darkslategray', line_width=1)
                ParametricPlot(u, branch, interval=(lower, bound), samples=120,
                               color='darkslategray', line_width=1)

            # Lines of constant t' and constant x' continue to the left/bottom
            # plot edges; the far endpoints are clipped at the top/right edges.
            moving_grid = []
            for i in ticks:
                # Solve ct' = gamma * (ct - v*x) at the left edge.
                start, end = (lower, i/gamma + v*lower), boost(upper, i)
                # Swap x and ct for the other family of gridlines.
                moving_grid.extend(((start, end), (start[::-1], end[::-1])))
            _segments(moving_grid, color='darkslategray', line_width=1, dash='2px,5px')

        ParametricPlot(u, u, interval=(0, upper*sqrt(2)),
                       color='gray', dash='10px,5px', name='light: x = ct', samples=2)
        ParametricPlot(*boost(u, 0), interval=(0, 7),
                       color='white', line_width=1.5, name="x' axis", samples=2)
        ParametricPlot(*boost(0, u), interval=(0, 7),
                       color='white', line_width=1.5, name="ct' axis", samples=2)
        ParametricPlot(*boost(length, u),
                       interval=(guide_start(length), 7), color='lightblue',
                       line_width=1.5, name="constant x' = length", samples=2)
        at_event_time = boost(u, time)
        ParametricPlot(*at_event_time,
                       interval=(guide_start(time), 7), color='lightgreen',
                       line_width=1.5, name="constant ct' = time", samples=2)
        ParametricPlot(*at_event_time, interval=(0, length), color='red',
                       line_width=1.5, name='pencil', samples=2)
        event_x, event_ct = boost(length, time)
        ListPlot([event_x], [event_ct], name='transformed event')

        # Moving-axis ticks run parallel to the other moving axis, following
        # the transformed coordinate grid. Keep their visible length constant.
        tick_half = 0.04
        dct = tick_half / sqrt(1 + v**2)
        dx = v*dct
        home_ticks, moving_ticks = [], []
        for i in ticks:
            home_ticks.extend((((i, -tick_half), (i, tick_half)),
                               ((-tick_half, i), (tick_half, i))))
            x, ct = boost(i, 0)
            start, end = (x - dx, ct - dct), (x + dx, ct + dct)
            moving_ticks.extend(((start, end), (start[::-1], end[::-1])))
        _segments(home_ticks, color='gray', line_width=1.5)
        _segments(moving_ticks, color='white', line_width=1.5)

        # Place home-frame labels beside the primed labels at rest, with a
        # slight offset along each axis so the labels do not overlap at v=0.
        for label, (x, ct), color in (
            ('x', (5.85, -0.2), 'gray'),
            ('ct', (-0.2, 5.85), 'gray'),
            ("x'", boost(5.5, -0.2), 'white'),
            ("ct'", boost(-0.2, 5.5), 'white'),
        ):
            ListPlot([x], [ct], text=[label], color=color, points=0, name=f'{label} label')

    return diagram
