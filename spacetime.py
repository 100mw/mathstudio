"""The Week 12 spacetime diagram, written with MathStudio plotting commands.

Students import mathstudio separately, then call SpacetimeDiagram()
in a VS Code or Colab notebook.
Coordinates use c = 1: the vertical coordinate is ct, and v means velocity/c.
"""

from mathstudio import Slider, Plot, ParametricPlot, ListPlot, sqrt, t, u

__all__ = ['SpacetimeDiagram']


def SpacetimeDiagram(*, length=0, time=0, v=0, grid=True,
                     width=700, height=700, show=True):
    """Display the lab diagram with live length, time, and velocity sliders.

    MathStudio is the fixed plotting library; no plotting-library argument
    is needed.
    length and time are the event's primed coordinates; v is the moving
    frame's velocity in units of c. Inputs set initial slider values.
    The returned Plot exposes .figure, .sliders, and .at(length=..., v=...).
    The initial window matches the PDF; events may move outside it.
    The diagram uses Plotly's SVG traces with live sliders and zoom/pan controls.
    """
    length = Slider('length', 0, 6, 0.125, length)
    time = Slider('time', 0, 6, 0.125, time)
    v = Slider('v', 0, 0.99, 0.01, v)

    # Inverse Lorentz transformation: primed event -> unprimed diagram.
    # Python uses ** for powers. v = 1 is excluded because gamma diverges.
    gamma = 1 / sqrt(1 - v**2)

    def X(x, t):
        return gamma * (x + v*t)

    def T(x, t):
        return gamma * (v*x + t)

    with Plot(x=(-0.5, 6.25), y=(-0.5, 6.25), width=width, height=height,
              xlabel='', ylabel='',
              show=show) as diagram:
        for axis in ('xaxis', 'yaxis'):
            diagram.layout[axis].update(
                zerolinecolor='gray', zerolinewidth=1.5, tick0=0, dtick=1,
                minor=dict(tick0=0, dtick=0.25, showgrid=grid,
                           gridcolor='#181818', gridwidth=0.5, ticks=''),
            )
        if grid:
            # Sample the exact hyperbolas directly instead of sending contour
            # grids on each redraw. Include the visible part below/left of zero.
            for i in range(1, 7):
                bound = sqrt(6.25**2 - i**2)
                ParametricPlot(sqrt(u**2 + i**2), u, variable=u, t=(-0.5, bound),
                               color='darkslategray', line_width=1)
                ParametricPlot(u, sqrt(u**2 + i**2), variable=u, t=(-0.5, bound),
                               color='darkslategray', line_width=1)

            # Lines of constant t' and constant x' continue to the left/bottom
            # plot edges; the far endpoints are clipped at the top/right edges.
            for i in range(1, 7):
                ParametricPlot(X(u, i), T(u, i), variable=u,
                               t=(-i*v - 0.5/gamma, 6.25),
                               color='darkslategray', line_width=1, dash='2px,5px',
                               samples=2)
                ParametricPlot(X(i, u), T(i, u), variable=u,
                               t=(-i*v - 0.5/gamma, 6.25),
                               color='darkslategray', line_width=1, dash='2px,5px',
                               samples=2)

        ParametricPlot(u, u, variable=u, t=(0, 6.25*sqrt(2)),
                       color='gray', dash='10px,5px', name='light: x = ct', samples=2)
        ParametricPlot(X(u, 0), T(u, 0), variable=u, t=(0, 7),
                       color='white', line_width=1.5, name="x' axis", samples=2)
        ParametricPlot(X(0, t), T(0, t), t=(0, 7),
                       color='white', line_width=1.5, name="ct' axis", samples=2)
        # Continue past the moving axes to the bottom/left plot boundaries.
        # The far endpoints extend beyond the top/right and are clipped to view.
        ParametricPlot(X(length, t), T(length, t),
                       t=(-length*v - 0.5/gamma, 7), color='lightblue',
                       line_width=1.5, name="constant x' = length", samples=2)
        ParametricPlot(X(u, time), T(u, time), variable=u,
                       t=(-time*v - 0.5/gamma, 7), color='lightgreen',
                       line_width=1.5, name="constant ct' = time", samples=2)
        ParametricPlot(X(u, time), T(u, time), variable=u, t=(0, length),
                       color='red', line_width=1.5, name='pencil', samples=2)
        ListPlot([X(length, time)], [T(length, time)],
                 color='cyan', points=10, name='transformed event')

        # Moving-axis ticks run parallel to the other moving axis, following
        # the transformed coordinate grid. Keep their visible length constant.
        def tick(x, y, dx, dy, color='white'):
            ListPlot([x - dx, x + dx], [y - dy, y + dy],
                     color=color, points=0, lines=True, line_width=1.5)

        tick_half = 0.04
        moving_tick_half = tick_half / sqrt(1 + v**2)
        for i in range(1, 7):
            tick(i, 0, 0, tick_half, color='gray')
            tick(0, i, tick_half, 0, color='gray')
            tick(X(i, 0), T(i, 0), v*moving_tick_half, moving_tick_half)
            tick(X(0, i), T(0, i), moving_tick_half, v*moving_tick_half)

        # Place home-frame labels beside the primed labels at rest, with a
        # slight offset along each axis so the labels do not overlap at v=0.
        label_size = 16
        ListPlot([5.85], [-0.2], text=['x'], text_size=label_size,
                 color='gray', points=0, name='x label')
        ListPlot([-0.2], [5.85], text=['ct'], text_size=label_size,
                 color='gray', points=0, name='ct label')

        # Transform the label centers from fixed primed-frame positions too.
        ListPlot([X(5.5, -0.2)], [T(5.5, -0.2)],
                 text=["x'"], text_size=label_size,
                 color='white', points=0, name="x' label")
        ListPlot([X(-0.2, 5.5)], [T(-0.2, 5.5)],
                 text=["ct'"], text_size=label_size,
                 color='white', points=0, name="ct' label")

    return diagram
