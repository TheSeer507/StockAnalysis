import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import datetime
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd



app = dash.Dash(__name__)
app.title = "Flattened Columns with yfinance"

app.layout = html.Div([
    html.H1("Stock Data (Flattened Columns)"),

    # Ticker Input
    html.Div([
        html.Label("Enter Ticker: "),
        dcc.Input(
            id='ticker-input',
            type='text',
            value='AAPL'  # default
        )
    ], style={"marginBottom": "20px"}),

    # Date Range
    html.Div([
        html.Label("Select Date Range: "),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 2, 1)
        )
    ], style={"marginBottom": "20px"}),

    # Submit Button
    html.Button('Submit', id='submit-button', n_clicks=0),

    html.Hr(),

    # Graph Output
    dcc.Graph(id='stock-graph'),

    # Error Message
    html.Div(id='error-message', style={"color": "red", "marginTop": "20px"})
])


@app.callback(
    Output('stock-graph', 'figure'),
    Output('error-message', 'children'),
    Input('submit-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def update_graph(n_clicks, ticker, start_date, end_date):
    # Initialize an empty figure and no error
    fig = go.Figure()
    error_msg = ""

    if not ticker:
        error_msg = "Please enter a ticker symbol."
        return fig, error_msg

    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Download data with group_by="ticker"
    df = yf.download([ticker], start=start_date, end=end_date, group_by="ticker")

    if df.empty:
        error_msg = f"No data returned for {ticker}."
        return fig, error_msg

    # Flatten the multi-index columns
    df.columns = df.columns.to_flat_index()
    df.columns = [f"{c[0]}_{c[1]}" for c in df.columns]

    # We expect columns like: AAPL_Open, AAPL_High, AAPL_Low, AAPL_Close, AAPL_Volume
    # Build a figure with a line chart for Close and bar chart for Volume
    close_col = f"{ticker}_Close"
    volume_col = f"{ticker}_Volume"

    if close_col not in df.columns:
        error_msg = f"Column '{close_col}' not found."
        return fig, error_msg

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[close_col].tolist(),
        mode='lines',
        name='Close Price'
    ))

    # Volume bars on secondary y-axis
    if volume_col in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[volume_col].tolist(),
            name='Volume',
            opacity=0.4,
            yaxis='y2'
        ))
    else:
        error_msg = f"Column '{volume_col}' not found. Volume might be missing."

    # Layout with secondary y-axis
    fig.update_layout(
        title=f"{ticker} from {start_date.date()} to {end_date.date()}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying='y', side='right'),
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    return fig, error_msg


if __name__ == '__main__':
    app.run_server(debug=True)

app = dash.Dash(__name__)
app.title = "Flattened Columns with yfinance"

app.layout = html.Div([
    html.H1("Stock Data (Flattened Columns)"),

    # Ticker Input
    html.Div([
        html.Label("Enter Ticker: "),
        dcc.Input(
            id='ticker-input',
            type='text',
            value='AAPL'  # default
        )
    ], style={"marginBottom": "20px"}),

    # Date Range
    html.Div([
        html.Label("Select Date Range: "),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2020, 2, 1)
        )
    ], style={"marginBottom": "20px"}),

    # Submit Button
    html.Button('Submit', id='submit-button', n_clicks=0),

    html.Hr(),

    # Graph Output
    dcc.Graph(id='stock-graph'),

    # Error Message
    html.Div(id='error-message', style={"color": "red", "marginTop": "20px"})
])


@app.callback(
    Output('stock-graph', 'figure'),
    Output('error-message', 'children'),
    Input('submit-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def update_graph(n_clicks, ticker, start_date, end_date):
    # Initialize an empty figure and no error
    fig = go.Figure()
    error_msg = ""

    if not ticker:
        error_msg = "Please enter a ticker symbol."
        return fig, error_msg

    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Download data with group_by="ticker"
    df = yf.download([ticker], start=start_date, end=end_date, group_by="ticker")

    if df.empty:
        error_msg = f"No data returned for {ticker}."
        return fig, error_msg

    # Flatten the multi-index columns
    df.columns = df.columns.to_flat_index()
    df.columns = [f"{c[0]}_{c[1]}" for c in df.columns]

    # We expect columns like: AAPL_Open, AAPL_High, AAPL_Low, AAPL_Close, AAPL_Volume
    # Build a figure with a line chart for Close and bar chart for Volume
    close_col = f"{ticker}_Close"
    volume_col = f"{ticker}_Volume"

    if close_col not in df.columns:
        error_msg = f"Column '{close_col}' not found."
        return fig, error_msg

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[close_col].tolist(),
        mode='lines',
        name='Close Price'
    ))

    # Volume bars on secondary y-axis
    if volume_col in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[volume_col].tolist(),
            name='Volume',
            opacity=0.4,
            yaxis='y2'
        ))
    else:
        error_msg = f"Column '{volume_col}' not found. Volume might be missing."

    # Layout with secondary y-axis
    fig.update_layout(
        title=f"{ticker} from {start_date.date()} to {end_date.date()}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying='y', side='right'),
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    return fig, error_msg


if __name__ == '__main__':
    app.run_server(debug=True)

