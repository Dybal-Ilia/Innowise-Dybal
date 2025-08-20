import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import logging
import datetime as dt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)


class Plotter:

    
    def plot_distributions(self, df):

        """Plots feature distributions"""

        for col in df.select_dtypes(include=[np.number]):
            plt.figure()
            plt.hist(df[col], bins = 100, range = (0, max(df[col])))
            plt.title(f"{col} distribution")
            plt.xlabel(f"{col}")
            plt.ylabel("Amount")
            plt.show() 
            logger.info(f"Distribution for {col} plotted successfully - {dt.datetime.now()}")

    
    def plot_linear_corr(self, df):

        """Plots correlation matrix based on linear correlations"""

        corr = df.select_dtypes(include=[np.number]).corr()
        plt.figure()
        plt.title("Correlation matrix")
        sns.heatmap(corr, annot=True, fmt=".3f")
        plt.show()
        logger.info(f"Correlation matrix plotted successfullt - {dt.datetime.now()}")


    def plot_timeseries(self, df):

        """Plots the whole timeseries"""

        plt.figure(figsize=(20, 10))
        df.sort_values(by="date", ascending = True)
        sns.lineplot(data=df, x="date", y="item_cnt_day")
        plt.show()
        logger.info(f"Timeseries plotted successfully - {dt.datetime.now()}")



    def plot_timeseries_decomposed(self, df):

        """Decomposes timeseries and plots decomposed components"""

        ts = df.groupby("date")["item_cnt_day"].sum()
        decomposed = seasonal_decompose(ts, model="additive", period=30)
        fig, axes =plt.subplots(3, 1, figsize = (15, 10), sharex=True)

        decomposed.trend.plot(ax=axes[0], title="trend")
        logger.info(f"Trend plotted successfully - {dt.datetime.now()}")

        decomposed.seasonal.plot(ax=axes[1], title = "seasonal")
        logger.info(f"Seasonality plotted successfully - {dt.datetime.now()}")

        decomposed.resid.plot(ax=axes[2], title="resid")
        logger.info(f"Residuals plotted successfully - {dt.datetime.now()}")

        plt.show()

