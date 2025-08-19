import opendatasets as od
from collections import defaultdict
import os
import pandas as pd


def donwload_data(url):
    od.download(url)
    dataset_name = url.split("/")[-1]
    dataframes = defaultdict()
    for dirname, _, filenames in os.walk(f'{os.getcwd()}/{dataset_name}'):
        for filename in filenames:
            dataframes[filename] = pd.read_csv(f"{dirname}/{filename}")
    return dataframes