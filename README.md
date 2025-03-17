
```markdown
# Animated Support/Resistance & Single Trendline (Short Segments)

## Overview

This application is a **Dash** (Plotly) web application designed to visualize trading candlestick data and multiple approaches for detecting and plotting support/resistance (S/R) levels. It also includes an animated mechanism to step or play through different segments of the data, and an option to draw a single best-fitting trendline based on pivot points.

### Key Functionalities

1. **Candlestick Chart**  
   - Displays a selected window of the loaded CSV data (`H4-Data-Modified.csv` in this example).
   - User can control how many bars (candles) are visible at once.

2. **Animation / Navigation**  
   - Four buttons:
     - **<< Backward**: Move the visible window backward by one bar.
     - **Forward >>**: Move the visible window forward by one bar.
     - **Play ▶**: Automatically step forward at a configurable interval.
     - **Pause ⏸**: Stop the stepping.
   - A slider **Animation Speed** (ms) controls how quickly the forward stepping proceeds.

3. **Multiple S/R Methods** (chosen via **tabs**):  
   - **Pivot & Fibonacci**: Standard daily pivot calculation (P, S1, R1, etc.) plus Fibonacci retracements over a recent lookback.  
   - **Local Extrema**: Detects local maxima/minima from a smoothed close array, then draws short horizontal lines between pivot points that share similar prices.  
   - **ML Clustering**: Identifies pivot points from mild smoothing, then clusters them using one of **k-means**, **hierarchical**, or **DBSCAN**, generating center lines as potential S/R levels.  
   - **Score-Based Weighted**: Scores each bar’s High/Low for repeated bounces and breaks, picking the top lines with positive scores.  
   - **Histogram / Market Profile**: Creates a histogram of recent closes, then selects bins with the highest frequency as S/R lines.

4. **Trendlines**  
   - Optionally enabled via a checkbox.  
   - Computes local maxima/minima, checks for lines that pass near at least 3 pivot points, and **only** plots the single best line.

### Installation & Setup

1. **Clone or Copy** this repository (or the single script) to your local machine.

2. **Install Dependencies**:
   ```bash
   pip install dash dash-bootstrap-components plotly pandas numpy scipy scikit-learn
   ```
   Ensure your Python environment (3.8+ recommended) has these libraries.

3. **CSV Data**:  
   - Place your trading data file (e.g., `H4-Data-Modified.csv`) in the same folder or adjust the path in the script accordingly.

4. **Run the App**:
   ```bash
   python app.py
   ```
   By default, Dash will serve at [http://127.0.0.1:8050](http://127.0.0.1:8050). Visit that URL in your browser.

### Usage

1. **Candles in view** slider sets how many bars are shown at once.
2. **Animation Controls**:
   - **Backward** / **Forward** to manually navigate one bar at a time.
   - **Play** / **Pause** to animate forward movement automatically.
   - **Animation Speed** slider to adjust the step interval in milliseconds.
3. **S/R Methods**:
   - Select one tab (Pivot & Fibonacci, Local Extrema, etc.) at a time.
   - Adjust parameters (sliders or radio buttons) in each tab.  
4. **Trendlines**:
   - Check “Plot Trendlines” to enable single-line detection.
   - Adjust “Trendline Tolerance” to allow for more/less pivot alignment.

### Notes & Extensibility

- The app demonstrates how to quickly switch among various S/R detection methods.
- Code uses **Dash** + **Plotly** for interactive, real-time updates.
- You can enhance each detection method with more robust logic, caching, or advanced parameter tuning for production scenarios.

---

**Enjoy exploring your trading data with multiple support/resistance and trendline detection approaches!**
```
