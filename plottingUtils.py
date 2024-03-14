import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import random

def random_color():
    return "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])


def plot_log_candlesticks(df, products, normalize = True, remove_gaps = True, log_scale = True):

    fig = make_subplots(rows=1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
    
    for product in products:
        if normalize:
            renorm_factor = df['Open'][product].iloc[0]
            fig.add_trace(go.Candlestick(x=df.index,
                                        open=df['Open'][product]/renorm_factor,
                                        high=df['High'][product]/renorm_factor,
                                        low=df['Low'][product]/renorm_factor,
                                        close=df['Close'][product]/renorm_factor,
                                        name = product,
                                        increasing_line_color= random_color(), decreasing_line_color= random_color()),
                            row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=df.index,
                                        open=df['Open'][product],
                                        high=df['High'][product],
                                        low=df['Low'][product],
                                        close=df['Close'][product],
                                        name = product),
                        row=1, col=1)
        if remove_gaps:
            begin_date, end_date = [df.iloc[0].name, df.iloc[-1].name]
            df = df.reindex(pd.date_range(begin_date, end_date, freq='D'))
            datebreaks = df['Close'][product][df['Close'][product].isnull()].index
            fig.update_xaxes(rangebreaks=[dict(values=datebreaks)])
        if log_scale:
            fig.update_yaxes(title_text="Stock price logarithmic scale", type="log")
        else:
            fig.update_yaxes(title_text="Stock price linear scale")
    fig.update_layout(height=800, width=1200)
    fig.show()

#temp
def fit_distribution(dist, data, bounds):
    res = sp.stats.fit(dist, data, bounds = bounds)
    return res, res.nllf()

def hist_with_fit(data, dists, bounds, nbins = 150):

    fig = make_subplots(rows=2, cols=1, shared_yaxes='all', specs=[[{'type': 'box'}], [{'type': 'histogram'}]])
    fig1 = px.histogram(data, nbins=nbins, marginal="box", histnorm="probability density")
    fig.add_trace(go.Box(fig1.data[1]), row=1, col=1)
    fig.add_trace(go.Histogram(fig1.data[0]), row=2, col=1)

    for dist, bound in zip(dists, bounds):
        res, logL = fit_distribution(dist, data, bounds = bound )
        
        fig2 = px.line(x=np.linspace(np.min(data)*1.1, np.max(data)*1.1, 500),
                        y=dist.pdf(np.linspace(np.min(data)*1.1, np.max(data)*1.1, 500), *res.params))
        fig2.update_traces(line_color=random_color())
        #fig2.update_traces(name=f"{dist.name} fit, logL = {logL:.2f}")
        fig.add_trace(go.Scatter(fig2.data[0]), secondary_y=False, row=2, col=1)

    fig.update_layout(yaxis=dict(domain=[0.8516, 1.0]),
                    yaxis2=dict(domain=[0.0, 0.8316],),
                    xaxis=dict(showticklabels=False),                  
                    )

    fig.show()
