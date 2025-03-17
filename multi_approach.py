import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from scipy.signal import argrelextrema, savgol_filter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import dash

# ------------------------------------------------------
# Load & Prepare Data
# ------------------------------------------------------
df = pd.read_csv("data/H4-Data-Modified.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Drop missing
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ------------------------------------------------------
# 1) Pivot Points & Fibonacci
# ------------------------------------------------------
def compute_pivot_points_fib(dff):
    """
    We do a simple daily pivot formula:
    P = (High + Low + Close) / 3
    R1 = 2P - Low
    S1 = 2P - High
    ...
    For Fibonacci, we identify the last major swing high/low in the dff and
    create lines for 0.236, 0.382, 0.5, 0.618, 1.0.
    """
    # For demonstration, let's assume we take the last bar for pivot calc
    last_row = dff.iloc[-1]
    H, L, C = last_row["High"], last_row["Low"], last_row["Close"]
    P = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (R1 - S1)
    S2 = P - (R1 - S1)

    pivot_lines = [("Pivot", P), ("R1", R1), ("S1", S1), ("R2", R2), ("S2", S2)]

    # For Fibonacci, pick a naive approach: last 50 bars => find swing high/low
    lookback = min(50, len(dff))
    sub = dff.iloc[-lookback:]
    swing_high = sub["High"].max()
    swing_low = sub["Low"].min()
    fib_levels = []
    if swing_high != swing_low:
        diff = swing_high - swing_low
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 1.0]
        for r in fib_ratios:
            level = swing_high - r * diff
            fib_levels.append(("Fib " + str(r), level))

    return pivot_lines, fib_levels

# ------------------------------------------------------
# 2) Local Max/Min (argrelextrema)
# ------------------------------------------------------
def detect_local_extrema(series, order=5):
    """
    Basic local max/min detection using argrelextrema from SciPy.
    'order' is the window on each side to consider.
    Returns indices of local maxima and minima.
    """
    arr = series.values
    # local maxima
    local_max_idx = argrelextrema(arr, np.greater, order=order)[0]
    # local minima
    local_min_idx = argrelextrema(arr, np.less, order=order)[0]
    return local_max_idx, local_min_idx

# ------------------------------------------------------
# 3) ML Clustering (KMeans, Hierarchical, DBSCAN)
# ------------------------------------------------------
def cluster_swing_points(prices, method="kmeans"):
    """
    Given a list/array of pivot prices, cluster them to find potential S/R.
    We'll do a simple approach returning the cluster centers or medians as lines.
    """
    X = np.array(prices).reshape(-1, 1)

    if len(X) < 2:
        return []

    # Choose method
    if method == "kmeans":
        k = min(5, len(X))  # Just a small default
        model = KMeans(n_clusters=k, random_state=42).fit(X)
        centers = model.cluster_centers_
        lines = sorted(centers.flatten().tolist())
    elif method == "hierarchical":
        # We'll pick a distance threshold or number of clusters
        # For demonstration, let's pick 5 clusters
        model = AgglomerativeClustering(n_clusters=min(5, len(X)))
        labels = model.fit_predict(X)
        lines = []
        for lab in np.unique(labels):
            cluster_vals = X[labels == lab].flatten()
            lines.append(np.median(cluster_vals))
        lines.sort()
    elif method == "dbscan":
        # Some default eps
        model = DBSCAN(eps=1.0, min_samples=2).fit(X)
        labels = model.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # noise
        lines = []
        for lab in unique_labels:
            cluster_vals = X[labels == lab].flatten()
            lines.append(np.median(cluster_vals))
        lines.sort()
    else:
        lines = []

    return lines

# ------------------------------------------------------
# 4) Score-Based Weighted (Simple Demonstration)
# ------------------------------------------------------
def score_based_lines(dff):
    """
    We take each bar's High and Low as potential lines, then give scores for
    how often price bounces near them vs. cuts them. VERY simplified version.
    We'll just pick the top ~5 lines with highest positive score.
    """
    candidate_lines = []

    # Collect high and low from each bar
    for i, row in dff.iterrows():
        candidate_lines.append(row["High"])
        candidate_lines.append(row["Low"])

    # Score them
    scores = {}
    for lvl in candidate_lines:
        score = 0
        # Check each bar for bounce or cut
        for i, row in dff.iterrows():
            # If the close is within X% of lvl, call it a bounce
            # If the candle body crosses lvl, call it a cut
            # Let's define a small threshold
            thresh = lvl * 0.001  # 0.1%
            hi, lo, op, cl = row["High"], row["Low"], row["Open"], row["Close"]

            if abs(lvl - cl) <= thresh:
                score += 1
            if (lvl < max(op, cl) and lvl > min(op, cl)):
                score -= 2  # cut by the body

        # Summarize
        if lvl not in scores:
            scores[lvl] = 0
        scores[lvl] += score

    # pick top 5 lines
    sorted_lines = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top5 = [lvl for lvl, sc in sorted_lines[:5] if sc > 0]
    top5.sort()
    return top5

# ------------------------------------------------------
# 5) Histogram / Market Profile Approach
# ------------------------------------------------------
def histogram_levels(series, bins=5):
    """
    Build histogram of the data, return lines at the highest-frequency bins.
    """
    if len(series) < 2:
        return []
    hist, edges = np.histogram(series, bins=bins)
    # find bin(s) with top frequencies
    max_freq = np.max(hist)
    lines = []
    for i, val in enumerate(hist):
        if val > 0.8 * max_freq:  # lines for bins near top freq
            # center of this bin
            bin_mid = (edges[i] + edges[i+1]) / 2
            lines.append(bin_mid)
    lines.sort()
    return lines

# ------------------------------------------------------
# Trendline detection from local maxima/minima
# ------------------------------------------------------
def detect_trendlines(xvals, yvals, pivot_idxs, tolerance=0.01):
    """
    Simple approach:
      - Take each pair of pivot points, form a line.
      - Check if any other pivot points lie 'close enough' to that line
        (within tolerance) => if 1 or more do, keep the line.
      - This is a naive demonstration. Real-world logic can be more robust.
    Returns: list of lines, each line is (slope, intercept, pivot_indices_in_line)
    """
    lines = []
    used = set()

    pivot_points = [(xvals[i], yvals[i]) for i in pivot_idxs]
    n_pivots = len(pivot_points)
    if n_pivots < 2:
        return lines

    for i in range(n_pivots):
        for j in range(i+1, n_pivots):
            x1, y1 = pivot_points[i]
            x2, y2 = pivot_points[j]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope*x1

            # check how many other pivot points lie close to this line
            match_count = 2
            included_indices = {i, j}
            for k in range(n_pivots):
                if k == i or k == j:
                    continue
                xk, yk = pivot_points[k]
                # distance from line = |y - (mx+b)| / sqrt(m^2 + 1)
                line_y = slope*xk + intercept
                dist = abs(yk - line_y)
                # we define tolerance as ratio to y or absolute
                if abs(dist) <= tolerance*abs(yk):
                    match_count += 1
                    included_indices.add(k)

            if match_count >= 3:
                # store line
                lines.append((slope, intercept, included_indices))

    # We might further filter or unify lines. Let's keep them all for demonstration.
    return lines

# ------------------------------------------------------
# Dash App Layout
# ------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.H2("Multi-Method Support/Resistance & Trendline Detection", style={"textAlign": "center"}),
    dcc.Graph(id="chart"),

    # Sliders & Controls
    dbc.Row([
        dbc.Col([
            html.Label("Candles in View:"),
            dcc.Slider(id="window-slider", min=50, max=500, step=50, value=200,
                       marks={i: str(i) for i in range(50, 501, 50)}),
        ], width=4),
        dbc.Col([
            html.Label("Local Extrema Order:"),
            dcc.Slider(id="extrema-order", min=1, max=20, step=1, value=5,
                       marks={i: str(i) for i in range(1,21)})
        ], width=4),
        dbc.Col([
            html.Label("Trendline Tolerance:"),
            dcc.Slider(id="trend-tolerance", min=0, max=0.05, step=0.005, value=0.01,
                       marks={i/100: str(i/100) for i in range(0,6)})
        ], width=4),
    ], className="my-3"),

    # Method selection
    dbc.Row([
        dbc.Col([
            html.Label("Choose S/R Method(s):"),
            dcc.Checklist(
                id="method-checklist",
                options=[
                    {"label": "Pivot & Fibonacci", "value": "pivotfib"},
                    {"label": "Local Max/Min", "value": "localextrema"},
                    {"label": "ML K-Means", "value": "kmeans"},
                    {"label": "ML Hierarchical", "value": "hierarchical"},
                    {"label": "ML DBSCAN", "value": "dbscan"},
                    {"label": "Score-Based Weighted", "value": "scorebased"},
                    {"label": "Histogram / Market Profile", "value": "histogram"}
                ],
                value=["localextrema"],  # default
                inline=True
            )
        ], width=8),
        dbc.Col([
            html.Label("Enable Trendlines?"),
            dcc.Checklist(
                id="trendline-checklist",
                options=[{"label": "Yes", "value": "trend"}],
                value=[]
            )
        ], width=4),
    ]),

    dcc.Store(id="current-start", data=0),
], fluid=True)


# ------------------------------------------------------
# Callbacks
# ------------------------------------------------------
@app.callback(
    Output("chart", "figure"),
    Input("window-slider", "value"),
    Input("extrema-order", "value"),
    Input("trend-tolerance", "value"),
    Input("method-checklist", "value"),
    Input("trendline-checklist", "value")
)
def update_chart(window_size, extrema_order, trend_tol, methods, trend_opts):
    """
    1) Crop data to chosen window size from the END by default
    2) For each selected method, compute lines and plot them
    3) If trendlines are enabled, detect from local max/min
    """
    # 1) Take last 'window_size' bars
    dff = df.iloc[-window_size:].reset_index(drop=True)
    fig = go.Figure()

    # Basic candlestick
    fig.add_trace(go.Candlestick(
        x=dff.index.astype(str),
        open=dff["Open"],
        high=dff["High"],
        low=dff["Low"],
        close=dff["Close"],
        name="Price"
    ))

    # 2) If user wants local maxima/min detection for other methods,
    #    let's store pivot points
    local_max_idx, local_min_idx = [], []
    pivot_prices = []
    if "localextrema" in methods or "kmeans" in methods or "hierarchical" in methods or "dbscan" in methods or "scorebased" in methods or "histogram" in methods:
        # We'll compute local extrema on smoothed close just to reduce noise
        # (like many do). This is optional, but often helpful:
        # For safety, if smooth_window > len(dff), clamp it
        smooth_w = min(len(dff)-1, 11) if len(dff) > 11 else len(dff)
        smooth_close = savgol_filter(dff["Close"], smooth_w, 3)
        sc_series = pd.Series(smooth_close)

        local_max_idx, local_min_idx = detect_local_extrema(sc_series, order=extrema_order)
        # We'll store pivot prices for possible clustering:
        pivot_prices = list(sc_series[local_max_idx]) + list(sc_series[local_min_idx])

    # 3) For each selected method, add lines
    #    We'll store them in a single list so we can plot them all at once
    all_lines = []

    # a) Pivot & Fibonacci
    if "pivotfib" in methods:
        pivot_pts, fibs = compute_pivot_points_fib(dff)
        # pivot_pts is list of tuples (name, level)
        # fibs is list of tuples (fib_label, level)
        for name, lvl in pivot_pts + fibs:
            all_lines.append((lvl, name))  # store

    # b) Local Max/Min
    if "localextrema" in methods:
        # Already found local_max_idx, local_min_idx
        # We'll just show lines at each local extremum
        # and also highlight the points
        for idx in local_max_idx:
            all_lines.append((sc_series[idx], "LocalMax"))
        for idx in local_min_idx:
            all_lines.append((sc_series[idx], "LocalMin"))

        # Plot the points themselves
        fig.add_trace(go.Scatter(
            x=[str(x) for x in local_max_idx],
            y=sc_series[local_max_idx],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Local Maxima"
        ))
        fig.add_trace(go.Scatter(
            x=[str(x) for x in local_min_idx],
            y=sc_series[local_min_idx],
            mode="markers",
            marker=dict(color="green", size=8),
            name="Local Minima"
        ))

    # c) ML Clustering
    # We'll cluster pivot_prices if present
    if any(m in methods for m in ["kmeans","hierarchical","dbscan"]) and len(pivot_prices) > 1:
        if "kmeans" in methods:
            lines = cluster_swing_points(pivot_prices, method="kmeans")
            for lvl in lines:
                all_lines.append((lvl, "K-Means"))
        if "hierarchical" in methods:
            lines = cluster_swing_points(pivot_prices, method="hierarchical")
            for lvl in lines:
                all_lines.append((lvl, "Hierarchical"))
        if "dbscan" in methods:
            lines = cluster_swing_points(pivot_prices, method="dbscan")
            for lvl in lines:
                all_lines.append((lvl, "DBSCAN"))

    # d) Score-Based Weighted
    if "scorebased" in methods:
        lines = score_based_lines(dff)
        for lvl in lines:
            all_lines.append((lvl, "ScoreBased"))

    # e) Histogram / Market Profile
    if "histogram" in methods:
        lines = histogram_levels(dff["Close"], bins=5)
        for lvl in lines:
            all_lines.append((lvl, "Histogram"))

    # 4) Plot all S/R lines from 'all_lines'
    #    We'll keep them in a dictionary for grouping by approach for the legend
    #    Key=approach, Value=list of levels
    from collections import defaultdict
    approach_levels = defaultdict(list)
    for lvl, label in all_lines:
        approach_levels[label].append(lvl)

    for label, line_vals in approach_levels.items():
        line_vals = sorted(set(line_vals))
        for lvl in line_vals:
            fig.add_hline(
                y=lvl,
                line_width=1,
                line_dash="dot",
                annotation_text=label,
                annotation_position="right",
                opacity=0.4
            )

    # 5) Trendlines if selected
    if "trend" in trend_opts:
        # We'll do naive trendlines from the same pivot points used above
        # For demonstration, let's use the local max/min approach
        sc_series = pd.Series(savgol_filter(dff["Close"], 5, 3))
        local_max_idx, local_min_idx = detect_local_extrema(sc_series, order=extrema_order)
        pivot_idxs = np.sort(np.concatenate([local_max_idx, local_min_idx]))
        lines = detect_trendlines(
            xvals=np.array(pivot_idxs, dtype=float),
            yvals=sc_series.values,
            pivot_idxs=pivot_idxs,
            tolerance=trend_tol
        )
        # Each line is (slope, intercept, pivot_indices_in_line)
        # We'll just plot them from the first to last pivot in that line
        for slope, intercept, pivs in lines:
            x_coords = sorted(pivs)
            x1 = x_coords[0]
            x2 = x_coords[-1]
            y1 = slope*x1 + intercept
            y2 = slope*x2 + intercept
            fig.add_trace(go.Scatter(
                x=[str(x1), str(x2)],
                y=[y1, y2],
                mode="lines",
                line=dict(width=1, dash="dash", color="magenta"),
                name="Trendline"
            ))

    # final layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        title=f"{window_size}-Bar View - Multi-Method S/R + Trendlines"
    )

    return fig


# ------------------------------------------------------
# Run
# ------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
