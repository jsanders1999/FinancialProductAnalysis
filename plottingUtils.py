import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots


def plot_log_candlesticks(df, products):
    for product in products:
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                            open=df['Open'][product],
                                            high=df['High'][product],
                                            low=df['Low'][product],
                                            close=df['Close'][product]
                                            )])

        begin_date, end_date = [df.iloc[0].name, df.iloc[-1].name]
        df = df.reindex(pd.date_range(begin_date, end_date, freq='D'))
        datebreaks = df['Close'][product][df['Close'][product].isnull()].index
        fig.update_xaxes(rangebreaks=[dict(values=datebreaks)])

        fig.update_yaxes(title_text="Stock price logarithmic scale", type="log")
        fig.show()
