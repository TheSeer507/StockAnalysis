import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import datetime
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
app.title = "Advanced Stock Analysis (No pandas_ta)"

# Layout
app.layout = dbc.Container([
    html.H1("Advanced Stock Analysis", className="mt-3 mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Enter Ticker:"),
            dcc.Input(id='ticker-input', type='text', value='AAPL')
        ], md=4),
        dbc.Col([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date.today()
            )
        ], md=6),
        dbc.Col([
            html.Br(),
            html.Button("Submit", id='submit-button', n_clicks=0, className="btn btn-primary")
        ], md=2),
    ]),

    html.Hr(),

    dbc.Tabs([
        dbc.Tab(label="Price Chart + Indicators", tab_id="tab-chart"),
        dbc.Tab(label="Fundamentals", tab_id="tab-fundamentals"),
        dbc.Tab(label="News (Placeholder)", tab_id="tab-news")
    ], id="tabs", active_tab="tab-chart"),

    # Content area
    html.Div(id="tab-content", className="p-4"),

    html.Div(id='error-message', style={"color": "red", "marginTop": "20px"})
], fluid=True)


@app.callback(
    Output('tab-content', 'children'),
    Output('error-message', 'children'),
    Input('tabs', 'active_tab'),
    Input('submit-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date')
)
def render_tab_content(active_tab, n_clicks, ticker, start_date, end_date):
    error_msg = ""

    if not ticker:
        return html.Div("Please enter a ticker."), "Ticker is empty."

    # Parse dates
    try:
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except Exception:
        return html.Div(), "Invalid date format."

    # On every submit, fetch data
    if n_clicks > 0:
        df = yf.download([ticker], start=start_date, end=end_date, group_by="ticker")
        if df.empty:
            return html.Div(), f"No data returned for {ticker}."

        # Flatten columns
        df.columns = df.columns.to_flat_index()
        df.columns = [f"{c[0]}_{c[1]}" for c in df.columns]

        close_col = f"{ticker}_Close"
        volume_col = f"{ticker}_Volume"
        if close_col not in df.columns:
            return html.Div(), f"Column '{close_col}' not found."

        # --- MANUAL INDICATORS (No pandas_ta) ---
        # 1) 20-day simple moving average
        df["SMA_20"] = df[close_col].rolling(window=20).mean()

        # 2) Bollinger Bands (Middle = SMA, Upper = SMA + 2*std, Lower = SMA - 2*std)
        rolling_std = df[close_col].rolling(window=20).std()
        df["BBM"] = df["SMA_20"]  # middle
        df["BBU"] = df["SMA_20"] + 2 * rolling_std
        df["BBL"] = df["SMA_20"] - 2 * rolling_std

        # Build content per tab
        if active_tab == "tab-chart":
            return build_chart_tab(df, ticker), error_msg
        elif active_tab == "tab-fundamentals":
            return build_fundamentals_tab(ticker), error_msg
        elif active_tab == "tab-news":
            return html.Div("News feature not implemented yet."), error_msg
        else:
            return html.Div("Invalid tab selected."), error_msg
    else:
        # No submit yet
        return html.Div("Enter a ticker and date range, then click Submit."), ""


def build_chart_tab(df, ticker):
    """
    Builds a figure with:
    - Close Price line
    - 20-day SMA
    - Bollinger Bands (BBL, BBM, BBU)
    - Volume on a secondary y-axis
    """
    close_col = f"{ticker}_Close"
    volume_col = f"{ticker}_Volume"

    fig = go.Figure()

    # Price line (Close)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[close_col],
        mode='lines',
        name='Close Price'
    ))

    # SMA 20
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["SMA_20"],
        mode='lines',
        line=dict(dash='dot', color='green'),
        name='SMA 20'
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["BBU"],
        line=dict(width=1, color='rgba(173,216,230,1.0)'),
        name='BB Upper'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["BBM"],
        line=dict(width=1, color='rgba(173,216,230,0.7)'),
        name='BB Middle'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["BBL"],
        line=dict(width=1, color='rgba(173,216,230,0.4)'),
        name='BB Lower'
    ))

    # Volume (bar trace on a secondary y-axis)
    if volume_col in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[volume_col],
            name='Volume',
            opacity=0.3,
            yaxis='y2'
        ))

    fig.update_layout(
        title=f"{ticker} Price Chart with Bollinger Bands",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )
    return dcc.Graph(figure=fig)


def build_fundamentals_tab(ticker):
    """
    Fetches basic fundamental info from yfinance's Ticker.info
    and displays it in a Dash Bootstrap table.
    """
    t = yf.Ticker(ticker)
    info = t.info  # Dictionary of fundamental data

    if not info:
        return html.Div("No fundamental data found.")

    fundamentals = [
        ("Ticker", ticker),
        ("Market Cap", info.get("marketCap")),
        ("Forward PE", info.get("forwardPE")),
        ("Trailing PE", info.get("trailingPE")),
        ("Dividend Yield", info.get("dividendYield")),
        ("Beta", info.get("beta")),
        ("52 Week High", info.get("fiftyTwoWeekHigh")),
        ("52 Week Low", info.get("fiftyTwoWeekLow")),
        ("Short Ratio", info.get("shortRatio")),
        ("Sector", info.get("sector")),
        ("Full Time Employees", info.get("fullTimeEmployees"))
    ]

    return html.Div([
        html.H4("Fundamental Data"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")]))] +
            [html.Tbody([
                html.Tr([
                    html.Td(param),
                    html.Td(str(value))
                ]) for param, value in fundamentals
            ])],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True
        )
    ])


if __name__ == "__main__":
    app.run_server(debug=True)
