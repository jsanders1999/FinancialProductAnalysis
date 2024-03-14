#Import standard libraries
import numpy as np
import scipy as sp
from skewt_scipy.skewt import skewt
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import seaborn as sns

from plottingUtils import hist_with_fit

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
    #res1 = (np.array(sp.integrate.quad(lambda x: (x/(1+risk_free_rate))*dist(*params).pdf(x/(1+risk_free_rate)), -1, 0))+
    #        np.array(sp.integrate.quad(lambda x: (-1/(1+risk_free_rate))*dist(*params).pdf(x/(1+risk_free_rate)), -np.inf, -1)))
            
    res2 =  np.array(sp.integrate.quad(lambda x: -dist(*params).cdf(x/(1+risk_free_rate)), -1, 0, limit = 1000))
    res2[0] = dist(*params).cdf(-1/(1+risk_free_rate)) + res2[0] + (-1/(1+risk_free_rate))*dist(*params).cdf(-1/(1+risk_free_rate))
    #print("normal integral:", res1)
    print("by parts:", res2)
    return res2[0]#sp.integrate.quad(lambda x: x/(1+risk_free_rate)*dist(*params).pdf(x/(1+risk_free_rate)), -np.inf, 0)[0]

def estimate_params(df, products, risk_free_rate = (1.033)**(1/365.25)-1):
    results = pd.DataFrame(columns=pd.Index(["mean", "std", "mean-std^2/2", "risk (R)", "mean-intrest (E)", "R/E"]))
    for product in products:
        mu = np.mean(df["dr"][product])
        sigma = np.std(df["dr"][product])
        E = mu-sigma**2/2 - risk_free_rate
        R = risk(sp.stats.norm, (mu, sigma), risk_free_rate)
        results.loc[product] = [mu, sigma, mu-sigma**2/2, R, E, R/E]
    return results

def fit_distribution(dist, data, bounds):
    res = sp.stats.fit(dist, data, bounds = bounds)
    return res, res.nllf()

def estimate_params_dist(df, products, dist, bounds, risk_free_rate = (1.033)**(1/365.25)-1):
    results = pd.DataFrame(columns=pd.Index(["mean", "std", "mean-std^2/2", "risk (R)", "mean-intrest (E)", "R/E"]))
    for product in products:
        #hist_with_fit(df["dr"][product].dropna(), [dist], [bounds], nbins = 150)
        res, logL = fit_distribution(dist, df["dr"][product].dropna(), bounds)
        print(res.params, logL)
        mu = dist(*res.params).mean()
        sigma = dist(*res.params).std()
        E = mu-sigma**2/2 - risk_free_rate
        R = risk(dist, res.params, risk_free_rate)
        results.loc[product] = [mu, sigma, mu-sigma**2/2, R, E, R/E]
    return results


