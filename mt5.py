import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from scipy.signal import argrelextrema, savgol_filter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# -----------------------
# NEW: MetaTrader 5 import
# -----------------------
import MetaTrader5 as mt5

# =========================================
# 0) MT5 Initialization
# =========================================
if not mt5.initialize():
    print("mt5.initialize() failed:", mt5.last_error())
    # You could optionally exit or show an error in the Dash UI
# If needed:
# logged_in = mt5.login(account=123456, password="ABC", server="YourBroker-Server")
# if not logged_in:
#     print("MT5 login failed:", mt5.last_error())

# --------------------------------------------------------------------------------
# Load Data (Your Provided CSV)
# --------------------------------------------------------------------------------
df = pd.read_csv("data/H4-Data-Modified.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --------------------------------------------------------------------------------
# 1) Pivot & Fibonacci
# --------------------------------------------------------------------------------
def compute_pivots_fib(dff, lookback=50):
    """
    Basic pivot & fibonacci calculations
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
            # If the line crosses the candle body, penalize it
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
    Very simple approach for demonstration.
    """
    pivot_points = [(xvals[i], yvals[i]) for i in pivot_idxs if i < len(xvals)]
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
                if abs(yk)>1e-12 and dist <= tolerance*abs(yk):
                    match_count +=1
                    included.add(k)
            if match_count>=3:
                lines.append((x1,y1,x2,y2))

    return lines

# ================================================================================
# NEW: RISK-BASED TRADE FUNCTIONS
# ================================================================================
def calculate_lot_size(symbol: str, risk_percent: float, stop_loss_points: float):
    """
    risk_value = (risk_percent / 100) * account_balance
    cost_per_lot = stop_loss_points * (tick_value_per_lot / tick_size)
    """
    acc_info = mt5.account_info()
    if acc_info is None:
        return 0.0
    account_balance = acc_info.balance

    risk_value = (risk_percent/100.0) * account_balance

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01
    tick_value_per_lot = symbol_info.trade_tick_value
    tick_size          = symbol_info.trade_tick_size
    if tick_size <= 0:
        tick_size = 1

    cost_per_lot = stop_loss_points * (tick_value_per_lot/tick_size)
    if cost_per_lot<=0:
        return 0.01

    raw_lot = risk_value / cost_per_lot

    # clamp to broker's min/max volume & step
    lot_min = symbol_info.volume_min
    lot_max = symbol_info.volume_max
    lot_step= symbol_info.volume_step

    steps_count = int(raw_lot / lot_step)
    final_lot   = steps_count*lot_step
    if final_lot < lot_min:
        final_lot = lot_min
    if final_lot > lot_max:
        final_lot = lot_max

    return round(final_lot,4)

def place_market_order(symbol: str, side: str, risk_p: float, sl_points: float, tp_points: float, comment: str):
    """
    side = 'BUY' or 'SELL'
    """
    # 1) Calculate lot
    lot = calculate_lot_size(symbol, risk_p, sl_points)
    if lot<=0:
        return f"Lot size <= 0. Check inputs. (Symbol={symbol}, risk%={risk_p}, SL={sl_points})"

    # 2) Find current price
    info_tick = mt5.symbol_info_tick(symbol)
    if info_tick is None:
        return f"Symbol {symbol} not found or no tick info."

    # 3) Decide price, SL, TP
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or not symbol_info.visible:
        # Try to select
        mt5.symbol_select(symbol, True)

    point = symbol_info.point
    if side.upper()=="BUY":
        price = info_tick.ask
        sl_price = price - sl_points*point
        tp_price = price + tp_points*point
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = info_tick.bid
        sl_price = price + sl_points*point
        tp_price = price - tp_points*point
        order_type = mt5.ORDER_TYPE_SELL

    # 4) Send the order
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 10,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return f"Trade failed. Retcode={result.retcode}, LastError={mt5.last_error()}"
    return f"Trade placed: {side} {symbol}, lot={lot}, ticket={result.order}"

# ================================================================================
# DASH APP
# ================================================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Main S/R method tabs
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

# Animation controls
animation_controls = dbc.Row([
    dbc.Col([
        dbc.Button("<< Backward", id="backward-btn", color="primary", className="mr-2"),
        dbc.Button("Forward >>",  id="forward-btn",  color="primary", className="ml-2"),
        dbc.Button("Play ▶",      id="play-btn",     color="success", className="ml-2"),
        dbc.Button("Pause ⏸",    id="pause-btn",    color="danger",  className="ml-2"),
    ], width=12)
])

# ---------------------------
# NEW: Trade Panel
# ---------------------------
trade_panel = dbc.Card([
    dbc.CardHeader("Trade Panel (MT5)"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Symbol:"),
                dcc.Input(id="inp-symbol", type="text", value="EURUSD", style={"width":"80px"})
            ], width=6),
            dbc.Col([
                html.Label("Risk %:"),
                dcc.Input(id="inp-risk", type="number", value=1.0, style={"width":"60px"})
            ], width=6),
        ], align="center"),

        dbc.Row([
            dbc.Col([
                html.Label("SL (points):"),
                dcc.Input(id="inp-sl", type="number", value=300, style={"width":"70px"}),
            ], width=6),
            dbc.Col([
                html.Label("TP (points):"),
                dcc.Input(id="inp-tp", type="number", value=750, style={"width":"70px"})
            ], width=6),
        ], align="center", className="mt-2"),

        dbc.Row([
            dbc.Col([
                html.Label("Comment:"),
                dcc.Input(id="inp-comment", type="text", value="Dash Trade", style={"width":"100%"})
            ], width=12),
        ], className="mt-2"),

        dbc.Row([
            dbc.Col([
                dbc.Button("Buy", id="btn-buy", color="success", className="mr-2", style={"width":"80px"}),
                dbc.Button("Sell", id="btn-sell", color="danger", className="ml-2", style={"width":"80px"}),
            ], width=12, className="mt-2")
        ]),

        html.Div(id="trade-result", className="mt-2 text-info")
    ])
], className="mt-3")

app.layout = dbc.Container([
    html.H3("Animated S/R Methods & Trendlines + Trade Panel", className="mt-2 mb-2 text-center"),

    dbc.Row([
        dbc.Col([
            # Left panel
            html.Div([
                html.Label("Candles in view:"),
                dcc.Slider(id="window-size", min=20, max=300, step=20, value=80,
                           marks={i: str(i) for i in range(20,301,40)}),
            ], className="my-2"),

            animation_controls,
            dcc.Interval(id="interval-component", interval=1000, n_intervals=0, disabled=True),
            dcc.Store(id="current-index", data=0),

            # S/R method tabs
            html.Label("Select S/R Method:"),
            tabs,

            # Trendlines
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

            # Add the trade panel
            trade_panel,

        ], width=4),

        dbc.Col([
            dcc.Loading(dcc.Graph(id="main-chart", style={"height":"80vh"}))
        ], width=8)
    ])
], fluid=True)


# --------------------------------------------------------------------------------
# 1) Animation / Navigation Callback
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 2) Play/Pause
# --------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------
# 3) Main Chart Callback
# --------------------------------------------------------------------------------
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
    # Slice data
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
        increasing_line_color="rgba(255,0,0,0.5)",
        decreasing_line_color="rgba(0,255,0,0.5)",
        opacity=0.5
    ))

    # Collect S/R lines
    sr_lines = []

    if len(dff) >= 2:
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
            if cluster_method in ["kmeans", "hierarchical"]:
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
        arr = savgol_filter(dff["Close"], 5, 3)
        arr_series = pd.Series(arr)
        max_idx = argrelextrema(arr_series.values, np.greater, order=5)[0]
        min_idx = argrelextrema(arr_series.values, np.less, order=5)[0]
        pivot_idxs = np.sort(np.concatenate([max_idx, min_idx]))
        pivot_idxs = pivot_idxs[pivot_idxs < len(arr)]
        lines = []
        if len(pivot_idxs)>1:
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
                        if abs(yk)>1e-12 and dist<=trend_tol*abs(yk):
                            match_count+=1
                            included.add(k_idx)
                    if match_count>=3:
                        lines.append((x1,y1,x2,y2))
        for (x1,y1,x2,y2) in lines:
            if x1>x2:
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
# 4) Handle Buy/Sell Button Click
# --------------------------------------------------------------------------------
@app.callback(
    Output("trade-result", "children"),
    Input("btn-buy", "n_clicks"),
    Input("btn-sell", "n_clicks"),
    State("inp-symbol", "value"),
    State("inp-risk", "value"),
    State("inp-sl", "value"),
    State("inp-tp", "value"),
    State("inp-comment", "value"),
    prevent_initial_call=True
)
def handle_trade(buy_clicks, sell_clicks, symbol, risk_p, sl_p, tp_p, comment):
    """
    If the user clicks 'Buy', we place a buy.
    If 'Sell', place a sell.
    Return the result message to trade-result div.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id=="btn-buy":
        side = "BUY"
    else:
        side = "SELL"

    # Place trade
    msg = place_market_order(symbol, side, float(risk_p), float(sl_p), float(tp_p), comment)
    return msg


# --------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------
if __name__=="__main__":
    app.run_server(debug=True)
