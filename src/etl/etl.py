import pandas as pd
import os
import sys
import logging
from decorators.exec_time import exec_time
import datetime as dt
from typing import Dict, List
from collections import defaultdict
import opendatasets as od
import numpy as np
from sklearn.impute import SimpleImputer
from dataloader.dataloader import donwload_data, merge_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETL:
    
    def __init__(self):
        self.default_pipeline = {
            "handle_missings": self._handle_missing_values,
            "handle_duplicates": self._handle_duplicates,
            "handle_negatives": self._handle_negatives,
            "process_categoricals": self._process_cat_names,
            "process_date": self._process_date_feature,
            "extend_features": self._extend_features
        }
        

    def extract(self, url:str):
        data = donwload_data(url=url)
        return merge_data(data)
        
    
    def _handle_missing_values(self, df:pd.DataFrame, numeric_strategy = "mean", categorical_strategy = "most_frequent"):



        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=numeric_strategy)
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        
        for col in cat_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=categorical_strategy)
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df

    def _handle_duplicates(self, df):
        return df.drop_duplicates()
    

    def _handle_negatives(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df.loc[df[col] < 0, col] = 0
        return df
    
    def _process_cat_names(self, df):
        cat_cols = cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.lower()
        return df
    

    def _process_date_feature(self, df):
        df["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")
        return df
    
    def _extend_features(self, df):
        df['city'] = [shop.strip("!").split(" ")[0] for shop in df["shop_name"]]
        ...
        return df
    

    def transform(self, df, pipeline:List[str] = None):
        if not pipeline:
            pipeline = self.default_pipeline

        for pipe in pipeline:
            if pipe in self.default_pipeline.keys():
                df = self.default_pipeline[pipe](df)
            else:
                raise ValueError(f"Can't process the pipe: {pipe}")
        return df


        


        

        
    




        


        



    




    

