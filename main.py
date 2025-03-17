import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from scipy.signal import argrelextrema, savgol_filter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import dash

# ---------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------
df = pd.read_csv("data/H4-Data-Modified.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------
# 0) Pattern-Change Helper
# ---------------------------------------------------------------------
def has_pattern_change(dff, idx1, idx2):
    sub = dff.iloc[idx1: idx2+1]
    if len(sub)<3:
        return False
    changes=0
    state=None
    for _, row in sub.iterrows():
        ctype= "buy" if row["Close"]>row["Open"] else "sell"
        if state is None:
            state= ctype
        elif ctype!= state:
            changes+=1
            state= ctype
        if changes>=2:
            return True
    return False

# ---------------------------------------------------------------------
# 1) Pivot & Fibonacci
# ---------------------------------------------------------------------
def compute_pivots_fib(dff, lookback=50):
    if len(dff)<2: return [], []
    last_row= dff.iloc[-1]
    H,L,C= last_row["High"], last_row["Low"], last_row["Close"]
    P= (H+L+C)/3
    R1= 2*P - L
    S1= 2*P - H
    R2= P + (R1-S1)
    S2= P - (R1-S1)
    pivot_lines= [("Pivot", P), ("R1",R1), ("S1",S1), ("R2",R2), ("S2",S2)]
    lb= min(len(dff), lookback)
    sub= dff.iloc[-lb:]
    hi= sub["High"].max()
    lo= sub["Low"].min()
    fibs=[]
    if hi> lo:
        diff= hi- lo
        for ratio in [0.236,0.382,0.5,0.618,1.0]:
            lvl= hi - ratio* diff
            fibs.append((f"Fib {ratio}", lvl))
    return pivot_lines, fibs

# ---------------------------------------------------------------------
# 2) Local Extrema
# ---------------------------------------------------------------------
def local_extrema_lines(dff, order=5, smooth_window=11):
    if len(dff)<2:
        return [],[],[],[]
    if smooth_window>= len(dff):
        smooth_window= max(3, len(dff)-1 if len(dff)%2==0 else len(dff))
    arr= savgol_filter(dff["Close"], smooth_window,3)
    max_idx= argrelextrema(arr, np.greater, order=order)[0]
    min_idx= argrelextrema(arr, np.less, order=order)[0]
    max_vals= arr[max_idx]
    min_vals= arr[min_idx]
    return max_vals, min_vals, max_idx, min_idx

def plot_local_extrema_segments(fig, dff, max_idx, min_idx, arr, price_tol=0.001):
    def unify_pivots(idx_array, price_array, color_type):
        used= [False]*len(idx_array)
        out= []
        for i in range(len(idx_array)):
            if used[i]:
                continue
            ix= idx_array[i]
            ival= price_array[i]
            group= [(ix, ival)]
            used[i]= True
            for j in range(i+1, len(idx_array)):
                if used[j]:
                    continue
                jx= idx_array[j]
                jval= price_array[j]
                if abs(jval- ival)<= price_tol* ival:
                    group.append((jx,jval))
                    used[j]= True
            if len(group)<2:
                continue
            xs= [g[0] for g in group]
            lx,rx= min(xs), max(xs)
            sub= dff.iloc[lx:rx+1] if rx>=lx else dff.iloc[rx:lx+1]
            if len(sub)==0:
                continue
            if color_type=="resist":
                line_price= sub["High"].max()
            else:
                line_price= sub["Low"].min()
            out.append((lx, line_price, rx, line_price, color_type))
        return out

    max_lines= unify_pivots(max_idx, arr[max_idx], "resist")
    min_lines= unify_pivots(min_idx, arr[min_idx], "support")
    for (lx,lval, rx,rval, typ) in (max_lines+ min_lines):
        x1,x2= (lx,rx) if lx<=rx else (rx,lx)
        color= "red" if typ=="resist" else "blue"
        fig.add_trace(go.Scatter(
            x=[str(x1), str(x2)],
            y=[lval, rval],
            mode="lines",
            line=dict(dash="dot", color=color),
            name=f"{typ} short-line"
        ))

# ---------------------------------------------------------------------
# 3) ML Clustering
# ---------------------------------------------------------------------
def cluster_swing_points(dff, method="kmeans", n_clusters=5, eps=1.0):
    if len(dff)<5:
        return []
    arr= savgol_filter(dff["Close"],5,3)
    s= pd.Series(arr)
    max_idx= argrelextrema(s.values, np.greater, order=5)[0]
    min_idx= argrelextrema(s.values, np.less, order=5)[0]
    pivot_prices= list(s[max_idx]) + list(s[min_idx])
    if len(pivot_prices)<2:
        return []
    X= np.array(pivot_prices).reshape(-1,1)
    lines=[]
    if method=="kmeans":
        k= min(n_clusters,len(X))
        model= KMeans(n_clusters=k, random_state=42).fit(X)
        centers= model.cluster_centers_.flatten()
        lines= sorted(centers.tolist())
    elif method=="hierarchical":
        k= min(n_clusters,len(X))
        model= AgglomerativeClustering(n_clusters=k)
        labs= model.fit_predict(X)
        for lb in np.unique(labs):
            cvals= X[labs==lb].flatten()
            lines.append(np.median(cvals))
        lines.sort()
    else:
        # dbscan
        model= DBSCAN(eps=eps, min_samples=2).fit(X)
        labs= model.labels_
        u= set(labs)
        if -1 in u:
            u.remove(-1)
        for lb in u:
            cvals= X[labs== lb].flatten()
            lines.append(np.median(cvals))
        lines.sort()
    return lines

# ---------------------------------------------------------------------
# 4) Score-Based Weighted
# ---------------------------------------------------------------------
def score_based_lines(dff, threshold=0.001, top_n=5):
    cands=[]
    for _, row in dff.iterrows():
        cands.append(row["High"])
        cands.append(row["Low"])
    scores={}
    for lvl in cands:
        sc=0
        for _, row in dff.iterrows():
            hi,lo,op,cl= row["High"], row["Low"], row["Open"], row["Close"]
            dist= abs(cl- lvl)
            if dist<= threshold* lvl:
                sc+=1
            if lvl< max(op,cl) and lvl> min(op,cl):
                sc-=2
        scores[lvl]= scores.get(lvl,0)+ sc
    srt= sorted(scores.items(), key=lambda x:x[1], reverse=True)
    out= [lvl for lvl,s in srt[:top_n] if s>0]
    return sorted(out)

# ---------------------------------------------------------------------
# 5) Histogram
# ---------------------------------------------------------------------
def histogram_levels(dff, bins=5):
    arr= dff["Close"].values
    if len(arr)<2:
        return []
    hist, edges= np.histogram(arr, bins=bins)
    mx= np.max(hist)
    lines=[]
    for i,hval in enumerate(hist):
        if hval> 0.5*mx:
            mid= (edges[i]+ edges[i+1])/2
            lines.append(mid)
    return sorted(lines)

# ---------------------------------------------------------------------
# Trendline utilities
# ---------------------------------------------------------------------
def is_valid_slope(slope, min_slope, max_slope):
    a= abs(slope)
    return (a>= min_slope) and (a<= max_slope)

def merge_similar_lines(lines, slope_tol=0.1, intercept_tol=0.5):
    merged=[]
    used=[False]*len(lines)
    for i in range(len(lines)):
        if used[i]: 
            continue
        x1,y1,x2,y2,slp,intc,mc,lt= lines[i]
        group= [i]
        for j in range(i+1, len(lines)):
            if used[j]: 
                continue
            _,_,_,_,slp2,intc2,mc2,lt2= lines[j]
            if abs(slp2- slp)< slope_tol and abs(intc2- intc)< intercept_tol:
                group.append(j)
        best_idx= i
        best_mc= mc
        for g in group:
            used[g]= True
            if lines[g][6]> best_mc:
                best_idx= g
                best_mc= lines[g][6]
        merged.append(lines[best_idx])
    return merged

def discard_overlapping_lines(lines_sorted):
    kept= []
    last_end= -1
    for (x1,y1,x2,y2,slp,intc,mc,lt) in lines_sorted:
        start= min(x1,x2)
        end= max(x1,x2)
        if start> last_end:
            kept.append((x1,y1,x2,y2,slp,intc,mc,lt))
            last_end= end
    return kept

# Key new function that ensures line is "outside" the candlesticks (no intersection).
# We'll use it for the base line too:
def shift_line_no_intersect(dff, xvals, slope, x1, x2, want="support"):
    """
    If want='support' => shift the line so liney <= candle_low for all bars in [x1..x2].
    If want='resistance' => shift so liney >= candle_high for all bars in [x1..x2].
    """
    idx1= min(x1,x2)
    idx2= max(x1,x2)
    if idx2>= len(dff): 
        return None
    sub= dff.iloc[idx1: idx2+1]
    if len(sub)<1:
        return None

    if want=="support":
        SHIFT= float("inf")
        # We want liney <= candle_low => candle_low - liney >=0 => liney= slope*x+ intercept
        # => intercept <= candle_low - slope*x => we pick the minimal of that for all bars
        # Actually, to ensure line is always below or equal to candle_low, we want the
        # maximum possible intercept that doesn't go above a candle's low? Actually we find min( candle_low - slope*x ) across the bars.
        # Then intercept= that min => ensures liney <= low for all bars.
        for i in range(idx1, idx2+1):
            c_lo= dff["Low"].iloc[i]
            liney= slope*xvals[i]
            # intercept <= c_lo - liney
            limit= c_lo - liney
            if limit< SHIFT:
                SHIFT= limit
        return SHIFT
    else:
        # want='resistance'
        SHIFT= float("-inf")
        # liney >= candle_high => intercept >= candle_high - slope*x
        for i in range(idx1, idx2+1):
            c_hi= dff["High"].iloc[i]
            liney= slope*xvals[i]
            limit= c_hi - liney
            if limit> SHIFT:
                SHIFT= limit
        return SHIFT

def line_score(x1, x2, match_count):
    span= abs(x2- x1)
    return match_count + 0.1*span

def detect_multiple_trendlines(
    dff,
    tolerance=0.01,
    min_slope=0.01,
    max_slope=10.0,
    slope_tol=0.1,
    intercept_tol=0.5,
    min_match_count=3,
    x_scale_factor=100.0
):
    if len(dff)<6:
        print("[DEBUG] Not enough candles.")
        return []
    arr= dff["Close"].values
    if len(dff)>=5:
        arr= savgol_filter(arr, 5,3)

    s= pd.Series(arr)
    max_idx= argrelextrema(s.values, np.greater, order=5)[0]
    min_idx= argrelextrema(s.values, np.less, order=5)[0]
    pivot_idxs= np.sort(np.concatenate([max_idx, min_idx]))
    pivot_idxs= pivot_idxs[pivot_idxs< len(s)]
    pivot_points= [(i, s[i]) for i in pivot_idxs]
    print("[DEBUG] pivot_points=", len(pivot_points))
    if len(pivot_points)<2:
        return []

    xvals= [i/x_scale_factor for i in range(len(dff))]

    lines=[]
    for i in range(len(pivot_points)):
        for j in range(i+1, len(pivot_points)):
            px1, py1= pivot_points[i]
            px2, py2= pivot_points[j]
            if px2== px1: 
                continue
            slope= (py2- py1)/(xvals[px2]- xvals[px1])
            intercept= py1 - slope*xvals[px1]
            match_count=2
            for k,(pidx,pval) in enumerate(pivot_points):
                if k in (i,j):
                    continue
                liney= slope*xvals[pidx] + intercept
                dist= abs(pval- liney)
                if dist<= tolerance*abs(pval):
                    match_count+=1
            if match_count>= min_match_count and is_valid_slope(slope, min_slope, max_slope):
                idx1= min(px1,px2)
                idx2= max(px1,px2)
                # Check pattern change
                if not has_pattern_change(dff, idx1, idx2):
                    continue
                # SHIFT base line => no candle intersection
                line_type= "support" if slope>0 else "resistance"
                SHIFT= shift_line_no_intersect(dff, xvals, slope, px1, px2, want=line_type)
                if SHIFT is None:
                    continue
                # new intercept
                new_int= SHIFT
                y1_new= slope*xvals[px1]+ new_int
                y2_new= slope*xvals[px2]+ new_int
                lines.append((px1,y1_new, px2,y2_new, slope,new_int, match_count, line_type))

    print("[DEBUG] Found lines BEFORE merging:", len(lines))
    if not lines:
        return []
    lines_merged= merge_similar_lines(lines, slope_tol, intercept_tol)
    print("[DEBUG] lines AFTER merging:", len(lines_merged))
    def line_xstart(L):
        (x1,_,x2,_,_,_,_,_)= L
        return min(x1,x2)
    lines_sorted= sorted(lines_merged, key=line_xstart)
    lines_no_overlap= discard_overlapping_lines(lines_sorted)
    print("[DEBUG] lines AFTER discard overlap:", len(lines_no_overlap))
    if not lines_no_overlap:
        return []

    # Now form parallel line => if base=support => parallel=resistance, etc.
    final_candidates= []
    for (px1,y1, px2,y2, slope,intc, mc, ltype) in lines_no_overlap:
        idx1= min(px1,px2)
        idx2= max(px1,px2)
        opp_type= "resistance" if ltype=="support" else "support"
        SHIFT2= shift_line_no_intersect(dff, xvals, slope, px1, px2, want=opp_type)
        if SHIFT2 is not None:
            # compute endpoints
            y1_opp= slope*xvals[idx1]+ SHIFT2
            y2_opp= slope*xvals[idx2]+ SHIFT2
            final_candidates.append((px1,y1, px2,y2, slope,intc, mc, ltype,(idx1,y1_opp, idx2,y2_opp)))
        else:
            final_candidates.append((px1,y1, px2,y2, slope,intc, mc, ltype,None))

    best_line= None
    best_score= float("-inf")
    for cand in final_candidates:
        (bx1,by1, bx2,by2, bslp,bint, bmc, bltype, par)= cand
        sc= line_score(bx1,bx2,bmc)
        if sc> best_score:
            best_score= sc
            best_line= cand
    if not best_line:
        return []
    # return [("base", bx1,by1,bx2,by2, slope, intercept, match_count, line_type, parallel?)]
    return [("base", *best_line)]

# ---------------------------------------------------------------------
# Build the Dash App
# ---------------------------------------------------------------------
app= Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

tabs= dbc.Tabs([
    dbc.Tab(label="Pivot & Fibonacci", tab_id="pivotfib"),
    dbc.Tab(label="Local Extrema", tab_id="localextrema"),
    dbc.Tab(label="ML Clustering", tab_id="mlcluster"),
    dbc.Tab(label="Score-Based Weighted", tab_id="scorebased"),
    dbc.Tab(label="Histogram / Market Profile", tab_id="histogram"),
], id="method-tabs", active_tab="localextrema")

animation_controls= dbc.Row([
    dbc.Col([
        dbc.Button("<< Backward", id="backward-btn", color="primary", className="mr-2"),
        dbc.Button("Forward >>", id="forward-btn", color="primary", className="ml-2"),
        dbc.Button("Play ▶", id="play-btn", color="success", className="ml-2"),
        dbc.Button("Pause ⏸", id="pause-btn", color="danger", className="ml-2"),
    ], width=12)
])

app.layout= dbc.Container([
    html.H3("Both 'Base' & 'Parallel' lines do not intersect the candles"),
    dbc.Row([
        dbc.Col([
            html.Label("Candles in view:"),
            dcc.Slider(id="window-size", min=20, max=300, step=20, value=80),
            animation_controls,
            dcc.Interval(id="interval-component", interval=500, n_intervals=0, disabled=True),
            dcc.Store(id="current-index", data=0),

            html.Label("Animation Speed (ms)"),
            dcc.Slider(id="animation-speed", min=100, max=2000, step=100, value=500),

            html.Label("Select S/R Method:"),
            tabs,

            # pivot fib
            html.Div([
                html.Label("PivotFib Lookback"),
                dcc.Slider(id="pivotfib-lookback", min=10, max=200, step=10, value=50),
            ], className="my-2"),

            # localextrema
            html.Div([
                html.Label("Extrema Order"),
                dcc.Slider(id="extrema-order", min=1, max=20, step=1, value=5),
                html.Label("Smooth Window"),
                dcc.Slider(id="extrema-smooth", min=3, max=51, step=2, value=11),
            ], className="my-2"),

            # ml cluster
            html.Div([
                html.Label("Clustering Method"),
                dcc.RadioItems(
                    id="cluster-method",
                    options=[{"label":"K-Means","value":"kmeans"},
                             {"label":"Hierarchical","value":"hierarchical"},
                             {"label":"DBSCAN","value":"dbscan"}],
                    value="kmeans",
                    inline=True),
                dcc.Slider(id="cluster-param", min=1, max=10, step=1, value=5)
            ], className="my-2"),

            # score-based
            html.Div([
                html.Label("Score Threshold (0.001=0.1%)"),
                dcc.Slider(id="score-threshold", min=0.0001, max=0.01, step=0.0001, value=0.001),
                html.Label("Score Top N lines"),
                dcc.Slider(id="score-topn", min=1, max=15, step=1, value=5),
            ], className="my-2"),

            # histogram
            html.Div([
                html.Label("Histogram Bins"),
                dcc.Slider(id="hist-bins", min=2, max=20, step=1, value=5),
            ], className="my-2"),

            html.Hr(),
            html.Label("Enable Multi Trendlines?"),
            dbc.Checklist(
                id="enable-trendlines",
                options=[{"label":"Both lines do not intersect the candles","value":"yes"}],
                value=[]
            ),
            html.Label("Trendline Tolerance"),
            dcc.Slider(id="trendline-tol", min=0.0, max=0.05, step=0.005, value=0.01),
            html.Label("Min Slope"),
            dcc.Slider(id="trendline-min-slope", min=0.0, max=5.0, step=0.1, value=0.01),
            html.Label("Max Slope"),
            dcc.Slider(id="trendline-max-slope", min=0.1, max=50.0, step=0.5, value=10.0),
            html.Label("Line Merge SlopeTol"),
            dcc.Slider(id="trendline-merge-slope-tol", min=0.0, max=1.0, step=0.01, value=0.1),
            html.Label("Line Merge InterceptTol"),
            dcc.Slider(id="trendline-merge-intercept-tol", min=0.0, max=5.0, step=0.1, value=0.5),
            html.Label("Min Match Count"),
            dcc.Slider(id="trendline-min-match-count", min=2, max=5, step=1, value=3),
            html.Label("Trendline X Scale"),
            dcc.Slider(id="trendline-xscale", min=1.0, max=1000.0, step=10.0, value=100.0),
        ], width=4),

        dbc.Col([
            dcc.Loading(dcc.Graph(id="main-chart", style={"height":"80vh"}))
        ], width=8)
    ])
], fluid=True)

@app.callback(
    Output("current-index","data"),
    [Input("forward-btn","n_clicks"), Input("backward-btn","n_clicks"), Input("interval-component","n_intervals")],
    [State("current-index","data"), State("window-size","value")]
)
def navigate(forward_clicks, backward_clicks, intervals, cur_idx, wsize):
    ctx= dash.callback_context
    if not ctx.triggered:
        return cur_idx
    trig= ctx.triggered[0]["prop_id"].split(".")[0]
    max_start= max(0, len(df)- wsize)
    if trig in ["forward-btn","interval-component"]:
        return min(cur_idx+1, max_start)
    elif trig=="backward-btn":
        return max(cur_idx-1, 0)
    return cur_idx

@app.callback(
    Output("interval-component","disabled"),
    [Input("play-btn","n_clicks"), Input("pause-btn","n_clicks")]
)
def toggle_animation(play, pause):
    ctx= dash.callback_context
    if not ctx.triggered:
        return True
    trig= ctx.triggered[0]["prop_id"].split(".")[0]
    return False if trig=="play-btn" else True

@app.callback(
    Output("interval-component","interval"),
    [Input("animation-speed","value")]
)
def anim_speed(ms):
    return ms

@app.callback(
    Output("main-chart","figure"),
    [
        Input("current-index","data"),
        Input("window-size","value"),
        Input("method-tabs","active_tab"),
        # pivot fib
        Input("pivotfib-lookback","value"),
        # localextrema
        Input("extrema-order","value"),
        Input("extrema-smooth","value"),
        # ml
        Input("cluster-method","value"),
        Input("cluster-param","value"),
        # score
        Input("score-threshold","value"),
        Input("score-topn","value"),
        # hist
        Input("hist-bins","value"),
        # lines
        Input("enable-trendlines","value"),
        Input("trendline-tol","value"),
        Input("trendline-min-slope","value"),
        Input("trendline-max-slope","value"),
        Input("trendline-merge-slope-tol","value"),
        Input("trendline-merge-intercept-tol","value"),
        Input("trendline-min-match-count","value"),
        Input("trendline-xscale","value"),
    ]
)
def update_chart(
    cur_idx, wsize, active_tab,
    fib_lookback,
    ex_order, ex_smooth,
    cluster_method, cluster_param,
    sc_thresh, sc_topn,
    hist_bins,
    enable_t, t_tol,
    t_min_slope, t_max_slope,
    t_slope_tol, t_int_tol,
    t_min_count,
    t_xscale
):
    end= min(cur_idx+wsize, len(df))
    dff= df.iloc[cur_idx:end].reset_index(drop=True)

    fig= go.Figure()
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
    if len(dff)<2:
        fig.update_layout(title="Not enough data.")
        return fig

    sr_lines=[]
    max_idx, min_idx= [], []

    # S/R method
    if active_tab=="pivotfib":
        piv, fibs= compute_pivots_fib(dff, fib_lookback)
        for (nm,val) in piv:
            if nm.startswith("S"):
                sr_lines.append((val,"support"))
            elif nm.startswith("R"):
                sr_lines.append((val,"resist"))
            else:
                sr_lines.append((val,"neutral"))
        for (nm,val) in fibs:
            sr_lines.append((val,"neutral"))

    elif active_tab=="localextrema":
        mxvals,mnvals,mxid,mnid= local_extrema_lines(dff, ex_order, ex_smooth)
        max_idx, min_idx= mxid,mnid

    elif active_tab=="mlcluster":
        lines=[]
        if cluster_method in ["kmeans","hierarchical"]:
            lines= cluster_swing_points(dff, cluster_method, int(cluster_param), 1.0)
        else:
            lines= cluster_swing_points(dff, "dbscan", 5, float(cluster_param))
        for lvl in lines:
            sr_lines.append((lvl,"neutral"))

    elif active_tab=="scorebased":
        lines= score_based_lines(dff, sc_thresh, sc_topn)
        for lvl in lines:
            sr_lines.append((lvl,"neutral"))

    elif active_tab=="histogram":
        lines= histogram_levels(dff, hist_bins)
        for lvl in lines:
            sr_lines.append((lvl,"neutral"))

    # localextrema => short lines
    if active_tab=="localextrema":
        arr= savgol_filter(dff["Close"], max(3, ex_smooth if ex_smooth<len(dff) else 3), 3)
        plot_local_extrema_segments(fig, dff, max_idx, min_idx, arr, price_tol=0.001)
    else:
        used= set()
        for (val, ttype) in sr_lines:
            if val in used: 
                continue
            used.add(val)
            c= "orange"
            if ttype=="support": c= "blue"
            elif ttype=="resist": c= "red"
            fig.add_hline(y=val, line_color=c, line_dash="dot", opacity=0.7)

    # multi lines
    if "yes" in enable_t and len(dff)>5:
        final_lines= detect_multiple_trendlines(
            dff,
            tolerance=t_tol,
            min_slope=t_min_slope,
            max_slope=t_max_slope,
            slope_tol=t_slope_tol,
            intercept_tol=t_int_tol,
            min_match_count=t_min_count,
            x_scale_factor=t_xscale
        )
        for item in final_lines:
            if item[0]!="base":
                continue
            # item => ("base", x1,y1,x2,y2, slope,intc, mc, line_type, parallel?)
            if len(item)==9:
                # no parallel
                _, bx1,by1, bx2,by2, slp,intc, mc, ltype = item
                if bx1> bx2:
                    bx1,bx2= bx2,bx1
                    by1,by2= by2,by1
                fig.add_trace(go.Scatter(
                    x=[str(bx1), str(bx2)],
                    y=[by1, by2],
                    mode="lines",
                    line=dict(width=2, dash="dash", color="magenta"),
                    name=f"{ltype} (mc={mc})"
                ))
            else:
                # length=10 => parallel line
                _, bx1,by1, bx2,by2, slp,intc, mc, ltype, par = item
                if bx1> bx2:
                    bx1,bx2= bx2,bx1
                    by1,by2= by2,by1
                fig.add_trace(go.Scatter(
                    x=[str(bx1), str(bx2)],
                    y=[by1, by2],
                    mode="lines",
                    line=dict(width=2, dash="dash", color="magenta"),
                    name=f"{ltype} base (mc={mc})"
                ))
                if par is not None:
                    (px1_opp, py1_opp, px2_opp, py2_opp)= par
                    if px1_opp> px2_opp:
                        px1_opp, px2_opp= px2_opp, px1_opp
                        py1_opp, py2_opp= py2_opp, py1_opp
                    fig.add_trace(go.Scatter(
                        x=[str(px1_opp), str(px2_opp)],
                        y=[py1_opp, py2_opp],
                        mode="lines",
                        line=dict(width=2, dash="dash", color="magenta"),
                        name=f"{ltype} parallel"
                    ))
                    # fill region if same x-range
                    if (bx1==px1_opp) and (bx2== px2_opp):
                        fig.add_trace(go.Scatter(
                            x=[str(bx1), str(bx2), str(bx2), str(bx1)],
                            y=[by1, by2, py2_opp, py1_opp],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=False,
                            name="channel region"
                        ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        title=f"Channel with Both Lines outside Candles, method={active_tab}",
        xaxis_rangeslider_visible=False
    )
    return fig

if __name__=="__main__":
    app.run_server(debug=True)
