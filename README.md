# Week 12: Lorentz transformation

Open `Week11.ipynb` in VS Code or upload it to Google Colab.
Students need only these commands after setup:

```python
import mathstudio
from spacetime import SpacetimeDiagram

diagram = SpacetimeDiagram()
```

The black plot uses SVG graphics with continuously updating `length`, `time`,
and `v` sliders, plus Plotly's zoom and pan controls. Keep the notebook kernel
running while using the sliders. Hyperbolas are sampled directly from their
equations, and straight lines use just their two endpoints to keep slider updates
small while retaining SVG graphics. The plotting and widget implementation
is in `mathstudio.py`. The mathematical construction is in `spacetime.py`.

## VS Code

1. Install the Python and Jupyter extensions.
2. In your notebook's Python environment, run
   `python -m pip install -r requirements.txt`. Install `ipykernel` if that
   environment needs a notebook kernel.
3. Open this folder and `Week11.ipynb`, select that Python kernel, and run the
   diagram cell.

## Google Colab

1. Upload `Week11.ipynb` using **File > Upload notebook**.
2. Run this setup cell to install the plotting dependencies and download the
   modules from GitHub:

   ```python
   %pip install -q -r https://raw.githubusercontent.com/100mw/mathstudio/main/requirements.txt

   !wget -q -O mathstudio.py https://raw.githubusercontent.com/100mw/mathstudio/main/mathstudio.py
   !wget -q -O spacetime.py https://raw.githubusercontent.com/100mw/mathstudio/main/spacetime.py
   ```

3. Run the diagram cell:

   ```python
   import mathstudio
   from spacetime import SpacetimeDiagram

   diagram = SpacetimeDiagram()
   ```

Rerun the setup cell whenever Colab starts a new runtime. The plot uses Colab's
normal Plotly renderer inside a built-in Jupyter output widget.

Colab manages its own `ipykernel` and `jupyter-client` versions.

## Coordinates and colors

This follows the supplied lab PDF's **inverse** Lorentz transformation:

```text
gamma = 1 / sqrt(1 - v**2)
X = gamma * (x' + v * ct')
ct = gamma * (v * x' + ct')
```

Units have `c = 1`, so `v` means velocity divided by the speed of light, and
`time` is the vertical primed coordinate `ct'`. The event starts at
`(x', ct') = (length, time)`. The slider bounds are 0 to 6 for length and time,
and 0 to **0.99** for velocity. The PDF's endpoint of 1 is excluded because
gamma is undefined there.

| Feature | Color |
| --- | --- |
| Home-frame axes and tick marks | Gray |
| Moving-frame axes and tick marks | White |
| Home-frame minor gridlines (three per unit interval) | Dark gray |
| Invariant hyperbolas and moving gridlines | Dark slate gray |
| Light ray x = ct | Gray, dashed |
| Constant x' and ct' through the event, extended to the plot edges | Light blue and light green, thin lines |
| Pencil from x' = 0 to x' = length at fixed ct' | Red |
| Transformed event | Cyan |

The initial view matches the PDF's range, -0.5 to 6.25 on both axes, with equal
scales so light has a 45-degree slope. Some events and tick marks leave this
window at high v. Use Plotly's zoom and pan controls to explore beyond the
initial view. The pencil endpoints are
simultaneous in the primed frame; its red segment is not a measurement of
length at fixed unprimed time.

```python
diagram = SpacetimeDiagram(length=2, time=1, v=0.6)
diagram = SpacetimeDiagram(grid=False)  # Hide both sets of gridlines.
```

## Write mathematical plots

```python
from mathstudio import *

v = Slider('v', 0, 0.99, 0.01)
gamma = 1 / sqrt(1 - v**2)

with Plot(x=(-0.5, 6.25), y=(-0.5, 6.25), xlabel='x', ylabel='ct'):
    ImplicitPlot(x**2 - y**2 - 1)  # Plot expression = 0.
    ParametricPlot(gamma*u, gamma*v*u, interval=(0, 7), color='green')
    ParametricPlot(gamma*v*u, gamma*u, interval=(0, 7), color='blue')
```

- Use `**` for powers, `#` for comments, and `range(1, 7)` for integers 1 through 6.
- `ImplicitPlot(Eq(x**2 - y**2, 1))` is equivalent to plotting the expression
  `x**2 - y**2 - 1`. Use `Eq`, since Python's `==` tests equality immediately.
- `ListPlot([x1, x2], [y1, y2])` plots points; expressions may contain sliders.
- `ParametricPlot` accepts `line_width`, `dash`, and `name`. The curve parameter
  defaults to `u`. Pass `variable=t` to use `t` as the parameter symbol. The
  interval defaults to `(0, 2*pi)` and may contain sliders, as in
  `interval=(0, length)`. Straight lines can use `samples=2`.
  Curves default to 600 samples.
- Every command in one `with Plot(...)` block shares axes and sliders. Outside a
  block, `ParametricPlot` displays a standalone plot.
- `SpacetimeDiagram(show=False)` builds without displaying. Its `.figure` and
  `.sliders` expose the live objects; `.at(v=0.6, length=2, time=1)` makes a static
  Plotly snapshot using initial values for omitted sliders.
- HTML exported from a static snapshot has no Python slider callbacks. Use the
  notebook for the live diagram.
