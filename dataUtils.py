import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

def dr(df, products):
    for product in products:
        
        close_data = df['Close'][product]
        available_closedata = df['Close'][product][~df['Close'][product].isnull()]

        dr = np.empty(len(close_data))
        dr[:] = np.nan

        #TODO: Make matrix operation
        j = 0
        for i, close_price in enumerate(close_data):
            if i == 0:
                j+=1
            elif j>=len(available_closedata):
                print(j, "is too large")
                break
            elif not np.isnan(close_price):
                dr[i]  = available_closedata.iloc[j]/available_closedata.iloc[j-1]-1
                j += 1
        df.insert(len(df.columns), ("dr", product), dr)

    return df

def risk(dist, params, risk_free_rate=0.0):
    #TODO: integration by parts
    return sp.integrate.quad(lambda x: x/(1+risk_free_rate)*dist(*params).pdf(x/(1+risk_free_rate)), -np.inf, 0)[0]

def estimate_params(df, products, risk_free_rate = (1.033)**(1/365.25)-1):
    results = pd.DataFrame(columns=pd.Index(["mean", "std", "mean-std^2/2", "risk (R)", "mean-intrest (E)", "R/E"]))
    for product in products:
        mu = np.mean(df["dr"][product])
        sigma = np.std(df["dr"][product])
        E = mu-sigma**2/2 - risk_free_rate
        R = risk(sp.stats.norm, (mu, sigma), risk_free_rate)
        results.loc[product] = [mu, sigma, mu-sigma**2/2, R, E, R/E]
    return results


