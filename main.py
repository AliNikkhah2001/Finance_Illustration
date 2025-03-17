import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from scipy.signal import argrelextrema, savgol_filter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import dash

# --------------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------------
df = pd.read_csv("data/H4-Data-Modified.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --------------------------------------------------------------------------------
# 1) Pivot Points & Fibonacci
# --------------------------------------------------------------------------------
def compute_pivots_fib(dff, lookback=50):
    """
    Basic pivot & fibonacci calculations:
    - P = (High + Low + Close) / 3
    - R1 = 2P - L, S1 = 2P - H, etc.
    For Fibonacci, we look for a swing high/low in the last 'lookback' bars.
    """
    last_row = dff.iloc[-1]
    H, L, C = last_row["High"], last_row["Low"], last_row["Close"]
    P = (H + L + C) / 3
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (R1 - S1)
    S2 = P - (R1 - S1)
    pivot_lines = [("Pivot", P), ("R1", R1), ("S1", S1), ("R2", R2), ("S2", S2)]

    lb = min(len(dff), lookback)
    sub = dff.iloc[-lb:]
    swing_high = sub["High"].max()
    swing_low = sub["Low"].min()
    fib_lines = []
    if swing_high > swing_low:
        diff = swing_high - swing_low
        for ratio in [0.236, 0.382, 0.5, 0.618, 1.0]:
            lvl = swing_high - ratio * diff
            fib_lines.append((f"Fib {ratio}", lvl))
    return pivot_lines, fib_lines

# --------------------------------------------------------------------------------
# 2) Local Extrema
# --------------------------------------------------------------------------------
def local_extrema_lines(dff, order=5, smooth_window=11):
    """
    Return local maxima & minima from a smoothed close array.
    """
    if smooth_window >= len(dff):
        smooth_window = max(3, len(dff)-1 if len(dff)%2==0 else len(dff))

    arr = savgol_filter(dff["Close"], smooth_window, 3)
    max_idx = argrelextrema(arr, np.greater, order=order)[0]
    min_idx = argrelextrema(arr, np.less, order=order)[0]
    max_lvls = list(arr[max_idx])
    min_lvls = list(arr[min_idx])
    return max_lvls, min_lvls, max_idx, min_idx

# --------------------------------------------------------------------------------
# 3) ML Clustering
# --------------------------------------------------------------------------------
def cluster_swing_points(dff, method="kmeans", n_clusters=5, eps=1.0):
    """
    Identify pivot points via mild smoothing, then cluster them (KMeans, etc.).
    Return cluster-center lines.
    """
    if len(dff) < 5:
        return []
    arr = savgol_filter(dff["Close"], 5, 3)
    s = pd.Series(arr)
    max_idx = argrelextrema(s.values, np.greater, order=5)[0]
    min_idx = argrelextrema(s.values, np.less, order=5)[0]
    pivot_prices = list(s[max_idx]) + list(s[min_idx])
    if len(pivot_prices)<2:
        return []

    X = np.array(pivot_prices).reshape(-1,1)
    lines = []
    if method == "kmeans":
        k = min(n_clusters, len(X))
        model = KMeans(n_clusters=k, random_state=42).fit(X)
        centers = model.cluster_centers_.flatten()
        lines = sorted(centers.tolist())
    elif method == "hierarchical":
        k = min(n_clusters, len(X))
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        for lab in np.unique(labels):
            cluster_vals = X[labels == lab].flatten()
            lines.append(np.median(cluster_vals))
        lines.sort()
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=2).fit(X)
        labels = model.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        for lab in unique_labels:
            cluster_vals = X[labels == lab].flatten()
            lines.append(np.median(cluster_vals))
        lines.sort()
    return lines

# --------------------------------------------------------------------------------
# 4) Score-Based Weighted
# --------------------------------------------------------------------------------
def score_based_lines(dff, threshold=0.001, top_n=5):
    """
    Each bar's high/low => candidate lines. Score them based on bounces vs breaks.
    Return top_n lines with positive score.
    """
    cands = []
    for _, row in dff.iterrows():
        cands.append(row["High"])
        cands.append(row["Low"])

    scores = {}
    for lvl in cands:
        sc = 0
        for _, row in dff.iterrows():
            hi, lo, op, cl = row["High"], row["Low"], row["Open"], row["Close"]
            dist = abs(cl - lvl)
            if dist <= threshold * lvl:
                sc += 1
            if lvl < max(op, cl) and lvl > min(op, cl):
                sc -= 2
        scores[lvl] = scores.get(lvl, 0) + sc

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_list = [lvl for lvl, s in sorted_scores[:top_n] if s>0]
    return sorted(top_list)

# --------------------------------------------------------------------------------
# 5) Histogram
# --------------------------------------------------------------------------------
def histogram_levels(dff, bins=5):
    """
    Build histogram of close prices, pick bin midpoints for top freq bins.
    """
    arr = dff["Close"].values
    if len(arr)<2:
        return []
    hist, edges = np.histogram(arr, bins=bins)
    mx = np.max(hist)
    lines = []
    for i, hval in enumerate(hist):
        if hval>0.5*mx:
            mid = (edges[i]+edges[i+1])/2
            lines.append(mid)
    return sorted(lines)

# --------------------------------------------------------------------------------
# Trendline detection
# --------------------------------------------------------------------------------
def detect_trendlines(xvals, yvals, pivot_idxs, tolerance=0.01):
    """
    Simple approach: for each pair of pivot points, if there's a 3rd pivot (or more)
    lying near that line, we keep it. Return lines from x1->x2 only.
    """
    pivot_idxs = pivot_idxs[pivot_idxs<len(xvals)]
    pivot_points = [(xvals[i], yvals[i]) for i in pivot_idxs]
    n_piv = len(pivot_points)
    lines = []
    if n_piv<2:
        return lines

    for i in range(n_piv):
        for j in range(i+1, n_piv):
            x1, y1 = pivot_points[i]
            x2, y2 = pivot_points[j]
            if x2==x1:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            match_count = 2
            included = {i,j}
            for k in range(n_piv):
                if k in included:
                    continue
                xk, yk = pivot_points[k]
                line_y = slope*xk + intercept
                dist = abs(yk - line_y)
                if dist <= tolerance/100*abs(yk):
                    match_count +=1
                    included.add(k)
            if match_count>=3:
                # line from (x1,y1)->(x2,y2) only
                lines.append((x1,y1,x2,y2))

    return lines


# --------------------------------------------------------------------------------
# Dash App Layout
# --------------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Main tabs for different S/R methods
tabs = dbc.Tabs([
    dbc.Tab(label="Pivot & Fibonacci", tab_id="pivotfib", children=[
        dbc.Row([
            dbc.Col([
                html.Label("Lookback (Fibonacci)"),
                dcc.Slider(id="pivotfib-lookback", min=10, max=200, step=10, value=50,
                           marks={i: str(i) for i in range(10,201,20)})
            ], width=12)
        ])
    ]),
    dbc.Tab(label="Local Extrema", tab_id="localextrema", children=[
        dbc.Row([
            dbc.Col([
                html.Label("Extrema Order"),
                dcc.Slider(id="extrema-order", min=1, max=20, step=1, value=5,
                           marks={i: str(i) for i in range(1,21)})
            ], width=12),
            dbc.Col([
                html.Label("Smooth Window"),
                dcc.Slider(id="extrema-smooth", min=3, max=51, step=2, value=11,
                           marks={i: str(i) for i in range(3,52,4)})
            ], width=12),
        ])
    ]),
    dbc.Tab(label="ML Clustering", tab_id="mlcluster", children=[
        dbc.Row([
            dbc.Col([
                html.Label("Method"),
                dcc.RadioItems(
                    id="cluster-method",
                    options=[{"label":"K-Means","value":"kmeans"},
                             {"label":"Hierarchical","value":"hierarchical"},
                             {"label":"DBSCAN","value":"dbscan"}],
                    value="kmeans",
                    inline=True
                )
            ], width=12),
            dbc.Col([
                html.Label("K / n_clusters / eps"),
                dcc.Slider(id="cluster-param", min=1, max=10, step=1, value=5,
                           marks={i:str(i) for i in range(1,11)})
            ], width=12)
        ])
    ]),
    dbc.Tab(label="Score-Based Weighted", tab_id="scorebased", children=[
        dbc.Row([
            dbc.Col([
                html.Label("Threshold (0.001=0.1%)"),
                dcc.Slider(id="score-threshold", min=0.0001, max=0.01, step=0.0001, value=0.001,
                           marks={0.0001:"0.0001", 0.002:"0.002",0.005:"0.005",0.01:"0.01"})
            ], width=12),
            dbc.Col([
                html.Label("Top N Lines"),
                dcc.Slider(id="score-topn", min=1, max=15, step=1, value=5,
                           marks={i:str(i) for i in range(1,16)})
            ], width=12)
        ])
    ]),
    dbc.Tab(label="Histogram / Market Profile", tab_id="histogram", children=[
        dbc.Row([
            dbc.Col([
                html.Label("Number of Bins"),
                dcc.Slider(id="hist-bins", min=2, max=20, step=1, value=5,
                           marks={i:str(i) for i in range(2,21,2)})
            ], width=12)
        ])
    ]),
], id="method-tabs", active_tab="pivotfib")

# Animation controls: Forward, Backward, Play, Pause, plus an Interval
animation_controls = dbc.Row([
    dbc.Col([
        dbc.Button("<< Backward", id="backward-btn", color="primary", className="mr-2"),
        dbc.Button("Forward >>", id="forward-btn", color="primary", className="ml-2"),
        dbc.Button("Play ▶", id="play-btn", color="success", className="ml-2"),
        dbc.Button("Pause ⏸", id="pause-btn", color="danger", className="ml-2"),
    ], width=12)
])

app.layout = dbc.Container([
    html.H3("Animated S/R Methods & Trendlines", className="mt-2 mb-2 text-center"),

    dbc.Row([
        dbc.Col([
            # Left panel
            # number of bars in the view
            html.Div([
                html.Label("Candles in view:"),
                dcc.Slider(id="window-size", min=20, max=300, step=20, value=80,
                           marks={i: str(i) for i in range(20,301,40)}),
            ], className="my-2"),

            # current-index store / animation
            animation_controls,
            dcc.Interval(id="interval-component", interval=1000, n_intervals=0, disabled=True),
            dcc.Store(id="current-index", data=0),

            html.Label("Select S/R Method:"),
            tabs,

            html.Div([
                html.Label("Enable Trendlines?"),
                dbc.Checklist(
                    id="trendline-checklist",
                    options=[{"label":"Plot Trendlines","value":"yes"}],
                    value=[]
                ),
                html.Label("Trendline Tolerance:"),
                dcc.Slider(id="trend-tol", min=0.0, max=0.05, step=0.005, value=0.01,
                           marks={i/100: str(i/100) for i in range(0,6)})
            ], className="my-2"),
        ], width=4),

        dbc.Col([
            dcc.Loading(dcc.Graph(id="main-chart", style={"height":"80vh"}))
        ], width=8)
    ])
], fluid=True)


# -----------------------------------------------------------------
# Animation / Navigation Callback
# -----------------------------------------------------------------
@app.callback(
    Output("current-index","data"),
    Input("forward-btn","n_clicks"),
    Input("backward-btn","n_clicks"),
    Input("interval-component","n_intervals"),
    State("current-index","data"),
    State("window-size","value")
)
def navigate(forward_clicks, backward_clicks, intervals, cur_idx, window_size):
    ctx = dash.callback_context
    if not ctx.triggered:
        return cur_idx
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    max_start = max(0, len(df)-window_size)

    if trigger_id=="forward-btn" or trigger_id=="interval-component":
        return min(cur_idx+1, max_start)
    elif trigger_id=="backward-btn":
        return max(cur_idx-1, 0)

    return cur_idx

# -----------------------------------------------------------------
# Play/Pause
# -----------------------------------------------------------------
@app.callback(
    Output("interval-component","disabled"),
    Input("play-btn","n_clicks"),
    Input("pause-btn","n_clicks")
)
def toggle_animation(play_clicks, pause_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return (False if trigger_id=="play-btn" else True)

# -----------------------------------------------------------------
# Main Chart Callback
# -----------------------------------------------------------------
@app.callback(
    Output("main-chart","figure"),
    [
        Input("current-index","data"),
        Input("window-size","value"),
        Input("method-tabs","active_tab"),
        Input("trendline-checklist","value"),
        Input("trend-tol","value"),

        # Pivot & Fib
        Input("pivotfib-lookback","value"),
        # Local Extrema
        Input("extrema-order","value"),
        Input("extrema-smooth","value"),
        # Clustering
        Input("cluster-method","value"),
        Input("cluster-param","value"),
        # Score-based
        Input("score-threshold","value"),
        Input("score-topn","value"),
        # Histogram
        Input("hist-bins","value"),
    ]
)
def update_chart(cur_idx,
                 window_size,
                 active_tab,
                 trend_opts,
                 trend_tol,
                 pivotfib_lookback,
                 extrema_order,
                 extrema_smooth,
                 cluster_method,
                 cluster_param,
                 score_threshold,
                 score_topn,
                 hist_bins):
    # Slice data from [cur_idx : cur_idx + window_size]
    end = min(cur_idx + window_size, len(df))
    dff = df.iloc[cur_idx:end].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dff.index.astype(str),
        open=dff["Open"],
        high=dff["High"],
        low=dff["Low"],
        close=dff["Close"],
        name="Price",
        # reduce candle opacity
        increasing_line_color="rgba(255,0,0,0.5)",
        decreasing_line_color="rgba(0,255,0,0.5)",
        opacity=0.5
    ))

    # We'll store lines: (price, 'support'/'resist'/'neutral')
    sr_lines = []

    # Apply the selected method
    if len(dff) < 2:
        # not enough data
        pass
    else:
        if active_tab=="pivotfib":
            pivs, fibs = compute_pivots_fib(dff, lookback=pivotfib_lookback)
            for name,val in pivs:
                if name.startswith("S"):
                    sr_lines.append((val,"support"))
                elif name.startswith("R"):
                    sr_lines.append((val,"resist"))
                else:
                    sr_lines.append((val,"neutral"))
            for name,val in fibs:
                sr_lines.append((val,"neutral"))

        elif active_tab=="localextrema":
            max_lvls, min_lvls, _, _ = local_extrema_lines(dff, order=extrema_order, smooth_window=extrema_smooth)
            for lvl in max_lvls:
                sr_lines.append((lvl,"resist"))
            for lvl in min_lvls:
                sr_lines.append((lvl,"support"))

        elif active_tab=="mlcluster":
            if cluster_method=="kmeans" or cluster_method=="hierarchical":
                lines = cluster_swing_points(dff, method=cluster_method, n_clusters=int(cluster_param), eps=1.0)
            else:
                # DBSCAN
                lines = cluster_swing_points(dff, method="dbscan", n_clusters=5, eps=float(cluster_param))
            for lvl in lines:
                sr_lines.append((lvl,"neutral"))

        elif active_tab=="scorebased":
            lines = score_based_lines(dff, threshold=score_threshold, top_n=score_topn)
            for lvl in lines:
                sr_lines.append((lvl,"neutral"))

        elif active_tab=="histogram":
            lines = histogram_levels(dff, bins=hist_bins)
            for lvl in lines:
                sr_lines.append((lvl,"neutral"))

    # Plot S/R lines
    used_set = set()
    for (val,typ) in sr_lines:
        if val in used_set:
            continue
        used_set.add(val)
        c = "orange"
        if typ=="support": c="blue"
        elif typ=="resist": c="red"
        fig.add_hline(y=val, line_color=c, line_dash="dot", opacity=0.7)

    # Trendlines
    if "yes" in trend_opts and len(dff)>5:
        # detect pivot points
        arr = savgol_filter(dff["Close"], 5, 3)
        arr_series = pd.Series(arr)
        max_idx = argrelextrema(arr_series.values, np.greater, order=5)[0]
        min_idx = argrelextrema(arr_series.values, np.less, order=5)[0]
        pivot_idxs = np.sort(np.concatenate([max_idx, min_idx]))
        pivot_idxs = pivot_idxs[pivot_idxs < len(arr)]

        lines = []
        if len(pivot_idxs)>1:
            lines = []
            # from each pair i-j, if there's a 3rd pivot that fits => a line
            for i in range(len(pivot_idxs)):
                for j in range(i+1, len(pivot_idxs)):
                    i_idx = pivot_idxs[i]
                    j_idx = pivot_idxs[j]
                    if j_idx == i_idx:
                        continue
                    x1, y1 = i_idx, arr[i_idx]
                    x2, y2 = j_idx, arr[j_idx]
                    slope = (y2-y1)/(x2-x1)
                    intercept = y1 - slope*x1
                    match_count=2
                    included={i_idx, j_idx}
                    for k_idx in pivot_idxs:
                        if k_idx in included:
                            continue
                        xk, yk = k_idx, arr[k_idx]
                        liney = slope*xk + intercept
                        dist = abs(yk - liney)
                        if dist<=trend_tol*abs(yk):
                            match_count+=1
                            included.add(k_idx)
                    if match_count>=3:
                        lines.append((x1,y1,x2,y2))

        # plot from x1->x2
        for (x1,y1,x2,y2) in lines:
            # ensure x1 < x2 for plotting
            if x1> x2:
                x1,x2 = x2,x1
                y1,y2 = y2,y1
            fig.add_trace(go.Scatter(
                x=[str(x1), str(x2)],
                y=[y1, y2],
                mode="lines",
                line=dict(width=1, dash="dash", color="magenta"),
                name="Trendline"
            ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        title=f"Index Range: {cur_idx} to {end} / Method: {active_tab}",
        xaxis_rangeslider_visible=False
    )
    return fig

# --------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------
if __name__=="__main__":
    app.run_server(debug=True)
