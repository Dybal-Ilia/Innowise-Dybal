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
    


    def __init__(self, map_pipes, filter_pipes):
        self.map_pipes = map_pipes
        self.filter_pipes = filter_pipes
        

    def apply_map(self, tables):
        return self._apply_pipes(tables, self.map_pipes)


    def apply_filter(self, tables):
        return self._apply_pipes(tables, self.filter_pipes)
    

    def _apply_pipes(self, tables, pipes):
        for table in tables:
            if table in pipes:
                pipe = pipes[table]
                tables[table] = self._apply_pipes(tables[table], pipe)
        return tables
    
    def _apply_pipe(self, table, pipe):
        for func in pipe:
            table = func(table)
        return table
    

    def apply_merge(self, tables):
        merged_tables_train = (
            tables['sales_train']
            .merge(tables['items'], on = 'item_id', how = 'left')
            .merge(tables['item_categories'], on = 'item_category_id', how = 'left')
            .merge(tables['shops'], on = 'shop_id', how = 'left')
            )
        
        merged_tables_test = (
            tables['test']
            .merge(tables['items'], on = 'item_id', how = 'left')
            .merge(tables['item_categories'], on = 'item_category_id', how = 'left')
            .merge(tables['shops'], on = 'shop_id', how = 'left')
        )

        return merged_tables_train, merged_tables_test



    def extract(self, url:str):

        """Downloads data from the provided url"""

        data = donwload_data(url=url)
        return merge_data(data)
        
    @exec_time
    def _handle_missing_values(self, df:pd.DataFrame, numeric_strategy = "mean", categorical_strategy = "most_frequent"):

        """Imputes missing values based on provided strategy"""

        numeric_strategies_available = ["mean", "meadin", "most_frequent"]
        categorical_strategies_available = ["most_frequent"]
        if not numeric_strategy in numeric_strategies_available or not categorical_strategy in categorical_strategies_available:
            raise ValueError("Invalid strategy")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=numeric_strategy)
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        
        for col in cat_cols:
            imputer = SimpleImputer(missing_values=np.nan, strategy=categorical_strategy)
            df[col] = imputer.fit_transform(df[[col]]).ravel()
        return df


    @exec_time
    def _handle_duplicates(self, df):

        """Drops fully duplicated rows"""

        return df.drop_duplicates()
    

    @exec_time
    def _handle_negatives(self, df):

        """Translates negative values to zeros"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df.loc[df[col] < 0, col] = 0
        return df
    


    @exec_time
    def _process_cat_names(self, df):

        """Translates categorical features' values to lowercase"""

        cat_cols = cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.lower()
        return df
    

    @exec_time
    def _process_date_feature(self, df):

        """Translated date to datetime"""

        df["date"] = pd.to_datetime(df["date"], format = "%d.%m.%Y")
        return df
    

    @exec_time
    def _extend_features(self, df):

        """Creates new features (yet to be developed)"""

        df['city'] = [shop.strip("!").split(" ")[0] for shop in df["shop_name"]]
        ...
        return df
    

    @exec_time
    def transform(self, df, pipeline:List[str] = None):

        """Applies data transfromations"""

        if not pipeline:
            pipeline = self.default_pipeline

        for pipe in pipeline:
            if pipe in self.default_pipeline.keys():
                df = self.default_pipeline[pipe](df)
                logger.info(f"{pipe} performed successfully - {dt.datetime.now()}")
            else:
                raise ValueError(f"Can't process the pipe: {pipe}")
        return df


        


        

        
    




        


        



    




    

