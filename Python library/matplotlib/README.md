# Matplotlib

Plotting fundamentals, a data-visualization cheat sheet, and a standalone script
that builds an advanced multi-panel dashboard covering plot types beyond the
everyday line/bar/scatter set.

| File | Description |
|------|-------------|
| [01_matplotlib_basics.ipynb](01_matplotlib_basics.ipynb) | Core plotting API: figures, axes, saving output to `assets/plot.png`. |
| [02_scatter_plot.ipynb](02_scatter_plot.ipynb) | Scatter plots — styling, markers, and color mapping. |
| [03_data_visualization_guide.ipynb](03_data_visualization_guide.ipynb) | A cheat-sheet style guide to choosing and building common chart types. |
| [weather_dashboard.py](weather_dashboard.py) | A standalone script generating a 16-panel dashboard (violin plots, heatmaps, contour plots, polar plots, 3D scatter, streamplots, and more) from synthetic weather data. |

## Data & assets

- `data/weather_data.csv` — sample daily weather data (temperature, humidity,
  rainfall) usable as an alternative input for visualization practice.
- `assets/plot.png` — a saved example output from the basics notebook.

## Running the dashboard script

```bash
python weather_dashboard.py
```

This generates synthetic weather data in-memory (no CSV required) and displays
a 4×4 grid of advanced plot types.
