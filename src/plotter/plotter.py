import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class Plotter:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    
    def plot_distributions(self):
        for col in self.df.select_dtypes(include=[np.number]):
            plt.figure()
            plt.hist(self.df[col], bins = 100)
            plt.title(f"{col} distribution")
            plt.xlabel(f"{col}")
            plt.ylabel("Amount")
            plt.show() 

    
    def plot_linear_corr(self):
        corr = self.df.select_dtypes(include=[np.number]).corr()
        plt.figure()
        plt.title("Correlation matrix")
        sns.heatmap(corr, annot=True, fmt=".3f")
        plt.show()


    def plot_timeseries(self):
        plt.figure(figsize=(20, 10))
        sns.lineplot(data=self.df, x="date", y="item_cnt_day")
        plt.show()


    def plot_timeseries_decomposed(self):
        ts = self.df.groupby("date")["item_cnt_day"].sum()
        decomposed = seasonal_decompose(ts, model="additive", period=30)
        fig, axes =plt.subplots(3, 1, figsize = (15, 10), sharex=True)
        decomposed.trend.plot(ax=axes[0], title="trend")
        decomposed.seasonal.plot(ax=axes[1], title = "seasonal")
        decomposed.resid.plot(ax=axes[2], title="resid")
        plt.show()

