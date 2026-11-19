"""Small mathematical plotting commands for VS Code notebooks and Colab.

Requires numpy, sympy, plotly, and ipywidgets. Combine commands
inside ``with Plot(...)`` to share axes and live sliders.
"""
import math
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
import sympy as sp
import plotly.graph_objects as go

__all__ = [
    'Slider', 'ParametricPlot', 'ImplicitPlot', 'ListPlot', 'Plot',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'exp', 'log', 'sqrt',
    'Abs', 'pi', 'E', 'symbols', 'Eq', 'x', 'y', 't', 'u'
]

sin, cos, tan = sp.sin, sp.cos, sp.tan
asin, acos, atan = sp.asin, sp.acos, sp.atan
exp, log, sqrt, Abs = sp.exp, sp.log, sp.sqrt, sp.Abs
pi, E, symbols = sp.pi, sp.E, sp.symbols
Eq = sp.Eq
x, y, t, u = sp.symbols('x y t u', real=True)

@dataclass(frozen=True)
class _SliderSpec:
    name: str
    minimum: float
    maximum: float
    step: float
    initial: float

_sliders = {}
_active_plot = ContextVar('mathstudio_plot', default=None)

_DARK_PANEL_CSS = """
<style>
/* VS Code forces a white background on the widget's outer output wrapper. */
.cell-output-ipywidget-background:has(.mathstudio-dark) {
    background: #000000 !important;
    padding: 0 !important;
    border: 0 !important;
}
.mathstudio-dark {
    color-scheme: dark;
    background: #000000;
    color: #eeeeee;
    box-sizing: border-box;
    border-radius: 0;
    --jp-layout-color0: #111111;
    --jp-layout-color1: #111111;
    --jp-layout-color2: #262626;
    --jp-layout-color3: #404040;
    --jp-ui-font-color1: #eeeeee;
    --jp-widgets-color: #eeeeee;
    --jp-widgets-label-color: #eeeeee;
    --jp-widgets-readout-color: #eeeeee;
    --jp-widgets-input-color: #eeeeee;
    --jp-widgets-input-background-color: #262626;
    --jp-widgets-slider-track-color: #404040;
    --jp-widgets-slider-handle-background-color: #60a5fa;
    --jp-widgets-slider-handle-border-color: #93c5fd;
}
.mathstudio-dark .widget-label,
.mathstudio-dark .widget-readout {
    color: #eeeeee;
    background: transparent;
}
.mathstudio-dark .noUi-target {
    background: #404040;
    border-color: #525252;
    box-shadow: none;
}
.mathstudio-dark .noUi-connect { background: #60a5fa; }
.mathstudio-dark .noUi-handle {
    background: #60a5fa;
    border-color: #93c5fd;
    box-shadow: none;
}
.mathstudio-dark .widget-output,
.mathstudio-dark .output,
.mathstudio-dark .output_area,
.mathstudio-dark .output_subarea {
    background: #000000;
    color: #eeeeee;
    margin: 0;
    padding: 0;
    border: 0;
    box-shadow: none;
}
</style>
"""


def Slider(name, minimum, maximum, step, value=None):
    """Register a slider and return a SymPy symbol for use in formulas."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError('Slider name must be a valid Python identifier')
    low, high, increment = map(lambda z: float(sp.N(z)), (minimum, maximum, step))
    if not all(map(math.isfinite, (low, high, increment))):
        raise ValueError('Slider bounds and step must be finite')
    if high <= low or increment <= 0:
        raise ValueError('Slider requires minimum < maximum and step > 0')
    initial = low if value is None else float(sp.N(value))
    if not math.isfinite(initial) or not low <= initial <= high:
        raise ValueError('Initial slider value must be within the bounds')
    sym = sp.Symbol(name, real=True)
    _sliders[name] = _SliderSpec(name, low, high, increment, initial)
    return sym


def _interval(value, label, symbolic=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f'{label} must be a two-element interval')
    bounds = tuple(sp.sympify(v) for v in value)
    if any(v.free_symbols for v in bounds):
        if symbolic:
            return bounds
        raise ValueError(f'{label} bounds must be numeric')
    numeric = tuple(float(v) for v in bounds)
    if not all(map(math.isfinite, numeric)) or numeric[0] >= numeric[1]:
        raise ValueError(f'{label} must be finite and increasing')
    return bounds


def _parameters(free):
    params = sorted(free, key=str)
    missing = [str(s) for s in params if str(s) not in _sliders]
    if missing:
        raise ValueError('Undeclared slider(s): ' + ', '.join(missing))
    return params


def _array(value, shape):
    a = np.broadcast_to(np.asarray(value, dtype=float), shape)
    return np.where(np.isfinite(a), a, np.nan)


@dataclass
class _Layer:
    trace: object
    evaluate: object
    parameters: list
    cached: object = None

    def data(self, values):
        # Static layers are computed only once.
        if self.parameters:
            return self.evaluate(values)
        if self.cached is None:
            self.cached = self.evaluate({})
        return self.cached


class Plot:
    """Combine plotting commands on one black canvas with shared sliders.

    Example::

        with Plot(x=(-1, 7), y=(-1, 7)) as diagram:
            ParametricPlot(t, t, t=(0, 7))
            ImplicitPlot(x**2 - y**2 - 1)

    ``diagram.figure`` is the live figure; ``diagram.sliders`` holds controls.
    Plots use Plotly's SVG traces with live sliders and zoom/pan controls.
    ``diagram.at(v=0.6)`` returns a static figure without changing the sliders.
    Use ``show=False`` to build a plot without displaying it.
    """

    def __init__(self, *, x=None, y=None, width=750, height=550,
                 title=None, xlabel='x', ylabel='y', legend=False, show=True):
        self.xrange = None if x is None else _interval(x, 'x')
        self.yrange = None if y is None else _interval(y, 'y')
        self.layers, self.parameters = [], {}
        self.figure, self.result, self.sliders = None, None, {}
        self._displayed = False
        self._show = show
        self.layout = dict(
            title=title, width=width, height=height, template='plotly_dark',
            paper_bgcolor='#000000', plot_bgcolor='#000000',
            font=dict(color='#eeeeee'), showlegend=legend,
            uirevision='mathstudio',
            xaxis=dict(title=xlabel, zeroline=True, zerolinecolor='#aaaaaa',
                       gridcolor='#202020', constrain='domain'),
            yaxis=dict(title=ylabel, scaleanchor='x', scaleratio=1,
                       zeroline=True, zerolinecolor='#aaaaaa', gridcolor='#202020'),
            margin=dict(l=55, r=25, t=60 if title else 25, b=50),
        )
        if self.xrange is not None:
            self.layout['xaxis']['range'] = [float(v) for v in self.xrange]
        if self.yrange is not None:
            self.layout['yaxis']['range'] = [float(v) for v in self.yrange]

    def __enter__(self):
        self._token = _active_plot.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        _active_plot.reset(self._token)
        if exc_type is None:
            self.display(show=self._show)

    def _add(self, layer):
        for symbol in layer.parameters:
            name = str(symbol)
            spec = _sliders[name]
            if name in self.parameters and self.parameters[name] != spec:
                raise ValueError(f'Slider {name} was redefined within this plot')
            self.parameters[name] = spec
        self.layers.append(layer)

    def at(self, **values):
        """Evaluate a snapshot at given slider values (defaults to initial values)."""
        unknown = values.keys() - self.parameters.keys()
        if unknown:
            raise ValueError('Unknown slider(s): ' + ', '.join(sorted(unknown)))
        current = {name: spec.initial for name, spec in self.parameters.items()}
        for name, value in values.items():
            value = float(value)
            spec = self.parameters[name]
            if not math.isfinite(value) or not spec.minimum <= value <= spec.maximum:
                raise ValueError(f'{name} must be between {spec.minimum} and {spec.maximum}')
            current[name] = value
        fig = go.Figure(layout=self.layout)
        for layer in self.layers:
            trace = layer.trace.to_plotly_json()
            trace.update(layer.data(current))
            fig.add_trace(trace)
        return fig

    def display(self, *, show=True):
        """Build the live display. Notebook widgets need a running Python kernel."""
        if self.result is None:
            base = self.at()
            if not self.parameters:
                self.figure = self.result = base
            else:
                try:
                    import ipywidgets as widgets
                except ImportError as exc:
                    raise ImportError(
                        'Live sliders need ipywidgets. Run '
                        '%pip install -r requirements.txt, then restart the kernel.'
                    ) from exc
                self.figure = base
                self.output = widgets.Output(layout=widgets.Layout(
                    margin='0', padding='0', border='0', width='100%',
                ))
                for name, spec in self.parameters.items():
                    self.sliders[name] = widgets.FloatSlider(
                        value=spec.initial, min=spec.minimum, max=spec.maximum,
                        step=spec.step, description=name, continuous_update=True,
                        readout_format='.3g', style={'description_width': '70px'},
                        layout=widgets.Layout(width='95%'),
                    )

                def update(change):
                    values = {name: control.value for name, control in self.sliders.items()}
                    changed = change['owner'].description
                    # Reuse static layers; update only slider-dependent traces.
                    with self.figure.batch_update():
                        for trace, layer in zip(self.figure.data, self.layers):
                            if any(str(s) == changed for s in layer.parameters):
                                trace.update(layer.data(values))
                    if self._displayed:
                        self._redraw()

                for control in self.sliders.values():
                    control.observe(update, names='value')
                stylesheet = widgets.HTML(value=_DARK_PANEL_CSS,
                                          layout=widgets.Layout(display='none'))
                self.result = widgets.VBox(
                    [stylesheet, *self.sliders.values(), self.output],
                    layout=widgets.Layout(margin='0', padding='0', border='0',
                                          width='100%', overflow='auto'),
                )
                self.result.add_class('mathstudio-dark')
        if show:
            if self.parameters:
                from IPython.display import display
                display(self.result)
                self._displayed = True
                self._redraw()
            else:
                self.figure.show()
        return self.result

    def _redraw(self):
        # Use the notebook's Plotly renderer, as in the earlier lab. FigureWidget
        # needs extra JavaScript modules that some notebook hosts cannot load.
        from IPython.display import clear_output
        with self.output:
            clear_output(wait=True)
            self.figure.show()

def _submit(layer, *, title=None, width=750, height=550, show=True):
    plot = _active_plot.get()
    if plot is not None:
        plot._add(layer)
        return layer
    plot = Plot(title=title, width=width, height=height)
    plot._add(layer)
    return plot.display(show=show)


def ParametricPlot(x, y, *, t=(0, 2 * pi), variable=None, samples=600,
                   title=None, color='#2563eb', width=750, height=550,
                   line_width=3, dash=None, name=None, show=True):
    """Plot (x, y) formulas, with sliders for their registered parameters.

    ``t=(start, end)`` can contain sliders, e.g. ``t=(0, length)``.
    ``variable`` defaults to t (or u when only u occurs). Inside a Plot
    block this adds a layer; outside it returns a figure or slider VBox.
    Use ``samples=2`` for straight lines; curved plots default to 600 points.
    """
    x, y = sp.sympify(x), sp.sympify(y)
    free = x.free_symbols | y.free_symbols
    if variable is None:
        variable = globals()['u'] if globals()['u'] in free and globals()['t'] not in free else globals()['t']
    elif isinstance(variable, str):
        variable = next((s for s in free if str(s) == variable), sp.Symbol(variable, real=True))
    if not isinstance(variable, sp.Symbol):
        raise ValueError('variable must be a symbol or its name')
    bounds = _interval(t, 't', symbolic=True)
    if any(variable in v.free_symbols for v in bounds):
        raise ValueError('Parameter bounds cannot depend on the curve variable')
    if not isinstance(samples, int) or not 2 <= samples <= 10000:
        raise ValueError('samples must be an integer between 2 and 10000')
    params = _parameters((free | set().union(*(v.free_symbols for v in bounds))) - {variable})
    fx, fy = [sp.lambdify([variable, *params], expr, modules='numpy') for expr in (x, y)]
    fb = sp.lambdify(params, bounds, modules='numpy')

    def evaluate(values):
        args = [values[str(p)] for p in params]
        low, high = map(float, fb(*args))
        # A zero-length pencil is a valid curve collapsed to one point.
        if not all(map(math.isfinite, (low, high))) or low > high:
            raise ValueError('The parameter interval must be finite and nondecreasing')
        grid = np.linspace(low, high, samples)
        with np.errstate(all='ignore'):
            px = _array(fx(grid, *args), grid.shape)
            py = _array(fy(grid, *args), grid.shape)
        valid = np.isfinite(px) & np.isfinite(py)
        return dict(x=np.where(valid, px, np.nan), y=np.where(valid, py, np.nan))

    trace = go.Scatter(mode='lines', line=dict(color=color, width=line_width, dash=dash),
                       name=name, showlegend=name is not None)
    return _submit(_Layer(trace, evaluate, params), title=title, width=width, height=height, show=show)


def ImplicitPlot(expression, *, x=None, y=None, variables=None, samples=220,
                 color='darkslategray', line_width=1, name=None, show=True):
    """Plot expression = 0, or Eq(left, right), on a rectangular grid.

    Use ** for powers and Eq(a, b) for an equation; Python's == tests equality.
    x and y specify numeric intervals, defaulting to the enclosing Plot ranges.
    """
    if expression is True or expression is False or expression in (sp.true, sp.false):
        raise ValueError('Use Eq(left, right) or left - right, rather than ==')
    expression = sp.sympify(expression)
    if isinstance(expression, sp.Equality):
        expression = expression.lhs - expression.rhs
    vx, vy = variables if variables is not None else (globals()['x'], globals()['y'])
    plot = _active_plot.get()
    xb = x if x is not None else (plot.xrange if plot is not None and plot.xrange is not None else (-6, 6))
    yb = y if y is not None else (plot.yrange if plot is not None and plot.yrange is not None else (-6, 6))
    xb, yb = _interval(xb, 'x'), _interval(yb, 'y')
    if not isinstance(samples, int) or not 20 <= samples <= 1000:
        raise ValueError('samples must be an integer between 20 and 1000')
    params = _parameters(expression.free_symbols - {vx, vy})
    fun = sp.lambdify([vx, vy, *params], expression, modules='numpy')
    gx, gy = np.linspace(*map(float, xb), samples), np.linspace(*map(float, yb), samples)
    xx, yy = np.meshgrid(gx, gy)

    def evaluate(values):
        with np.errstate(all='ignore'):
            z = _array(fun(xx, yy, *[values[str(p)] for p in params]), xx.shape)
        return dict(x=gx, y=gy, z=z)

    trace = go.Contour(contours=dict(start=0, end=0, size=1, coloring='none'),
                       autocontour=False, showscale=False, hoverinfo='skip',
                       line=dict(color=color, width=line_width),
                       name=name, showlegend=name is not None)
    return _submit(_Layer(trace, evaluate, params), show=show)


def ListPlot(x, y, *, color='cyan', points=10, lines=False, line_width=2,
             text=None, text_size=16, name=None, show=True):
    """Plot coordinate lists, with optional text at the same live positions.

    Set ``points=0`` with ``text`` to draw labels without markers.
    Coordinates may include symbolic slider expressions.
    """
    xs, ys = tuple(map(sp.sympify, x)), tuple(map(sp.sympify, y))
    if len(xs) != len(ys) or not xs:
        raise ValueError('ListPlot needs equally sized, nonempty coordinate lists')
    params = _parameters(set().union(*(v.free_symbols for v in (*xs, *ys))))
    fun = sp.lambdify(params, (xs, ys), modules='numpy')

    def evaluate(values):
        px, py = fun(*[values[str(p)] for p in params])
        return dict(x=_array(px, (len(xs),)), y=_array(py, (len(ys),)))

    mode = 'lines+markers' if lines and points else 'lines' if lines else 'markers'
    if text is not None:
        mode = (mode + '+text') if lines or points else 'text'
    trace = go.Scatter(mode=mode, marker=dict(color=color, size=points),
                       line=dict(color=color, width=line_width),
                       text=text, textfont=dict(color=color, size=text_size),
                       name=name, showlegend=name is not None)
    return _submit(_Layer(trace, evaluate, params), show=show)
