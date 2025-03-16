import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
import dash

# Load Data
df = pd.read_csv('data/H4-Data-Modified.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Remove days with no trading data
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Robust ZigZag Cluster Levels Algorithm
def zigzag_cluster_levels(data, peak_percent_delta=0.05, merge_distance=None, merge_percent=0.02):
    prices = data['Close'].values
    pivots = [prices[0]]

    for price in prices[1:]:
        if abs(price - pivots[-1]) / pivots[-1] >= peak_percent_delta:
            pivots.append(price)

    pivots = np.array(pivots).reshape(-1, 1)

    if len(pivots) < 2:
        return np.array([])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=merge_distance or merge_percent * np.mean(pivots),
        linkage='average'
    )

    clustering.fit(pivots)

    levels = pd.Series(pivots.flatten()).groupby(clustering.labels_).median().values
    return levels

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.H1("Dynamic Candlestick with Support/Resistance", style={'textAlign': 'center', 'marginTop': 20}),

    dcc.Graph(id="candlestick-chart"),

    dbc.Row([
        dbc.Button("<< Backward", id="backward-btn", color="primary", className="mr-2"),
        dbc.Button("Forward >>", id="forward-btn", color="primary", className="ml-2"),
        dbc.Button("Play ▶", id="play-btn", color="success", className="ml-2"),
        dbc.Button("Pause ⏸️", id="pause-btn", color="danger", className="ml-2"),
    ], className="my-3"),

    dcc.Interval(id="interval-component", interval=500, n_intervals=0, disabled=True),

    dbc.Row([
        dbc.Label("Candles in view:"),
        dcc.Slider(id='candle-slider', min=10, max=200, step=10, value=120,
                   marks={i: str(i) for i in range(10, 201, 10)}, tooltip={"placement": "bottom", "always_visible": True}),
    ], className="my-3"),

    dbc.Row([
        dbc.Label("Smooth Window Length:"),
        dcc.Slider(id='smooth-slider', min=3, max=51, step=2, value=11,
                   marks={i: str(i) for i in range(3, 52, 2)}, tooltip={"placement": "bottom", "always_visible": True}),
    ], className="my-3"),

    dcc.Store(id='current-index', data=0),
], fluid=True)

@app.callback(
    Output("candlestick-chart", "figure"),
    Input("current-index", "data"),
    Input("candle-slider", "value"),
    Input("smooth-slider", "value")
)
def update_chart(index, window_size, smooth_window):
    start = max(0, index)
    end = min(len(df), start + window_size)
    dff = df.iloc[start:end].reset_index(drop=True)

    if smooth_window >= len(dff):
        smooth_window = len(dff) - 1 if len(dff) % 2 == 0 else len(dff)

    dff['Smoothed_Close'] = savgol_filter(dff['Close'], smooth_window, 3)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dff.index,
        open=dff['Open'],
        high=dff['High'],
        low=dff['Low'],
        close=dff['Close'],
        name='Price'
    ))

    fig.add_trace(go.Scatter(x=dff.index, y=dff['Smoothed_Close'], mode='lines', name='Smoothed Close', line=dict(color='cyan')))

    levels = zigzag_cluster_levels(dff)
    if levels.size > 0:
        for level in levels:
            fig.add_hline(y=level, line_dash="dot", line_color="yellow", opacity=0.5)

    fig.update_layout(title='Candlestick with Support/Resistance and Smoothed Close',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark',
                      hovermode='x unified',
                      xaxis=dict(type='category'))

    return fig

@app.callback(
    Output('current-index', 'data'),
    Input('forward-btn', 'n_clicks'),
    Input('backward-btn', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    State('current-index', 'data'),
    State('candle-slider', 'value')
)
def navigate(forward_clicks, backward_clicks, intervals, current_index, window_size):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_index
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'forward-btn' or trigger_id == 'interval-component':
        return min(current_index + 1, len(df) - window_size)
    elif trigger_id == 'backward-btn':
        return max(current_index - 1, 0)

    return current_index

@app.callback(
    Output('interval-component', 'disabled'),
    Input('play-btn', 'n_clicks'),
    Input('pause-btn', 'n_clicks'),
)
def toggle_animation(play_clicks, pause_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return False if trigger_id == 'play-btn' else True

if __name__ == '__main__':
    app.run_server(debug=True)
