import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Load Data
df = pd.read_csv('/mnt/data/H4-Data-Modified.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Calculate indicators
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD Signal'] = macd.macd_signal()
bbands = BollingerBands(close=df['Close'])
df['bb_upper'] = bbands.bollinger_hband()
df['bb_lower'] = bbands.bollinger_lband()

# Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.H1("Advanced Trading Dashboard", style={'textAlign': 'center', 'marginTop': 20}),
    dcc.Graph(id="candlestick-chart"),

    dbc.Row([
        dbc.Col([
            html.Label('Select Chart Window:'),
            dcc.RangeSlider(
                id='range-slider',
                min=0,
                max=len(df)-1,
                step=1,
                value=[len(df)-100, len(df)-1],
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ])
    ], style={'marginTop': 30}),

    dbc.Row([
        dbc.Col([
            dbc.Checklist(
                options=[
                    {'label': 'RSI', 'value': 'RSI'},
                    {'label': 'MACD', 'value': 'MACD'},
                    {'label': 'Bollinger Bands', 'value': 'BBANDS'},
                ],
                value=['RSI', 'MACD'],
                id="indicator-checklist",
                inline=True,
            ),
        ], width="auto", style={'marginTop': 20}),
    ])
], fluid=True)

@app.callback(
    Output("candlestick-chart", "figure"),
    [Input("range-slider", "value"), Input("indicator-checklist", "value")]
)
def update_chart(range_vals, indicators):
    dff = df.iloc[range_vals[0]:range_vals[1]]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dff['Date'],
        open=dff['Open'],
        high=dff['High'],
        low=dff['Low'],
        close=dff['Close'],
        name='Price'
    ))

    # Add selected indicators
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['RSI'], name='RSI', yaxis='y2'))

    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['MACD'], name='MACD', yaxis='y3'))
        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['MACD Signal'], name='MACD Signal', yaxis='y3'))

    if 'BBANDS' in indicators:
        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['bb_upper'], name='BB Upper', line={'dash':'dot'}))
        fig.add_trace(go.Scatter(x=dff['Date'], y=dff['bb_lower'], name='BB Lower', line={'dash':'dot'}))

    # Layout with multiple axes
    fig.update_layout(
        title='Interactive Candlestick with Indicators',
        yaxis=dict(title='Price'),
        yaxis2=dict(title='RSI', overlaying='y', side='right', position=0.95),
        yaxis3=dict(title='MACD', anchor='free', overlaying='y', side='right', position=1.0),
        xaxis_rangeslider_visible=True,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(x=0, y=1.1, orientation="h")
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
