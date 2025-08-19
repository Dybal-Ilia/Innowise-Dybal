import pandas as pd
import os
import sys
import logging
from decorators.exec_time import exec_time
import datetime as dt
from typing import Dict
from collections import defaultdict
import opendatasets as od
import numpy as np
from sklearn.impute import SimpleImputer
from dataloader.dataloader import donwload_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETL:
    
    def __init__(self, url:str, numeric_missings_strategy = "median", cat_missings_strategy = "most_frequent", pipes:list = None):
        self.url = url
        self.dataset_name = url.split('/')[-1]
        self.numeric_missings_strategy = numeric_missings_strategy
        self.cat_missings_strategy = cat_missings_strategy
        self.df = None
        self.default_pipeline = {
            "handle_missings": self._handle_missing_values,
            "handle_duplicates": self._handle_duplicates,
            "handle_negatives": self._handle_negatives,
            "process_categoricals": self._process_cat_names,
            "process_date": self._process_date_feature,
            "extend_features": self._extend_features
        }
        self.pipes = pipes if pipes else list(self.default_pipeline.keys()) 
        


    def extract(self):
        dataframes = donwload_data(url=self.url)
        df = dataframes["sales_train.csv"].merge(dataframes["items.csv"], on='item_id', how='left').merge(dataframes["item_categories.csv"], on='item_category_id', how='left').merge(dataframes["shops.csv"], on='shop_id', how='left')
        self.df = df
        return self.df
    

    def _handle_missing_values(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=self.numeric_missings_strategy)
            self.df[col] = imputer.fit_transform(self.df[[col]]).ravel()
        
        for col in cat_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=self.cat_missings_strategy)
            self.df[col] = imputer.fit_transform(self.df[[col]]).ravel()
        return self.df

    def _handle_duplicates(self):
        return self.df.drop_duplicates()
    

    def _handle_negatives(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df.loc[self.df[col] < 0, col] = 0
        return self.df
    
    def _process_cat_names(self):
        cat_cols = cat_cols = self.df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            self.df[col] = self.df[col].astype(str).str.lower()
        return self.df
    

    def _process_date_feature(self):
        self.df["date"] = pd.to_datetime(self.df["date"], format = "%d.%m.%Y")
        return self.df
    
    def _extend_features(self):
        self.df['city'] = [shop.strip("!").split(" ")[0] for shop in self.df["shop_name"]]
        ...
        return self.df
    

    def transform(self):
        for pipe in self.pipes:
            if pipe in self.default_pipeline.keys():
                self.df = self.default_pipeline[pipe]()
            else:
                raise ValueError(f"Can't process the pipe: {pipe}")
        return self.df


        


        

        
    




        


        



    




    

