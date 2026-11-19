# Week 12: Lorentz transformation

Open `Lorentz transformation.ipynb` in VS Code or upload it to Google Colab.
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
small while retaining SVG graphics. The plotting and widget
implementation stays in `mathstudio.py`; the mathematical construction is in
`spacetime.py`. The student script imports `mathstudio` explicitly to make the
plotting library visible. `SpacetimeDiagram` always uses MathStudio and takes
no plotting-library argument. Since Python modules have separate namespaces,
`spacetime.py` also imports the MathStudio commands it uses.
`lorentz_transformation.py` is a minimal launch script for
`%run lorentz_transformation.py` in a notebook cell.

## VS Code

1. Install the Python and Jupyter extensions.
2. Open this folder and `Lorentz transformation.ipynb`.
3. Select a Python kernel, run the notebook's installation cell once, and restart
   the kernel if prompted. Continue with the diagram cell.

Use the notebook editor or a Jupyter Interactive Window for sliders. Running
`python lorentz_transformation.py` in a terminal does not provide a widget host.
The terminal command `python -m pip install -r requirements.txt` is an alternative
to the installation cell; use the same Python environment as the notebook kernel.

## Google Colab

1. Upload `Lorentz transformation.ipynb` using **File > Upload notebook**.
2. Run the installation cell.
3. Run the upload cell and select `mathstudio.py` and `spacetime.py` from this
   folder. They must be uploaded again after Colab deletes or resets the runtime.
4. Run the diagram cell. The plot uses Colab's normal Plotly renderer inside
   a built-in Jupyter output widget.

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

The optional notebook section shows a short construction using only equations
and plotting commands. The full PDF translation is in `spacetime.py`.

```python
from mathstudio import *

v = Slider('v', 0, 0.99, 0.01)
gamma = 1 / sqrt(1 - v**2)

with Plot(x=(-0.5, 6.25), y=(-0.5, 6.25), xlabel='x', ylabel='ct'):
    ImplicitPlot(x**2 - y**2 - 1)  # Plot expression = 0.
    ParametricPlot(gamma*u, gamma*v*u, variable=u, t=(0, 7), color='green')
    ParametricPlot(gamma*v*t, gamma*t, t=(0, 7), color='blue')
```

- Use `**` for powers, `#` for comments, and `range(1, 7)` for integers 1 through 6.
- `ImplicitPlot(Eq(x**2 - y**2, 1))` is equivalent to plotting the expression
  `x**2 - y**2 - 1`. Use `Eq`, since Python's `==` tests equality immediately.
- `ListPlot([x1, x2], [y1, y2])` plots points; expressions may contain sliders.
- `ParametricPlot` accepts `line_width`, `dash`, and `name`. Its `t` interval may
  contain sliders, such as `t=(0, length)`. Use `variable=u` to select a parameter.
  Straight lines can use `samples=2`; curves default to 600 samples.
- Every command in one `with Plot(...)` block shares axes and sliders. Outside a
  block, `ParametricPlot` still displays a standalone plot, as in the prior lab.
- `SpacetimeDiagram(show=False)` builds without displaying. Its `.figure` and
  `.sliders` expose the live objects; `.at(v=0.6, length=2, time=1)` makes a static
  Plotly snapshot using initial values for omitted sliders.
- HTML exported from a static snapshot has no Python slider callbacks. Use the
  notebook for the live diagram.

## Instructor checks

From this folder, after installing the requirements:

```bash
python -m unittest discover -s tests -v
```

Tests check the lab geometry, Lorentz invariant, collapsed pencil at length zero,
slider callbacks, symbolic parameter bounds, and the earlier Lissajous example.
Plots use a regular Plotly Figure inside an `ipywidgets.Output`, as in the earlier
Lissajous lab. Slider changes redraw the SVG plot through the notebook's Plotly
renderer, without FigureWidget or anywidget. If you previously loaded the
FigureWidget version, restart the kernel and rerun the notebook to load the
updated scripts.
