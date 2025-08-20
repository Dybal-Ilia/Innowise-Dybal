import opendatasets as od
from collections import defaultdict
import os
import pandas as pd
from typing import Dict


def donwload_data(url):
    od.download(url)
    dataset_name = url.split("/")[-1]
    dataframes = defaultdict()

    for dirname, _, filenames in os.walk(f'{os.getcwd()}/{dataset_name}'):
        for filename in filenames:
            dataframes[filename.split(".")[0]] = pd.read_csv(f"{dirname}/{filename}")

    return dataframes

def merge_data(dataframes: Dict[str, pd.DataFrame]):
        df = dataframes["sales_train"].merge(dataframes["items"], on='item_id', how='left').merge(dataframes["item_categories"], on='item_category_id', how='left').merge(dataframes["shops"], on='shop_id', how='left')
        return df




