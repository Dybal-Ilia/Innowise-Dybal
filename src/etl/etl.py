import pandas as pd
import os
import sys
import logging
from decorators.exec_time import exec_time
import datetime as dt
from typing import Dict, List, Tuple
from collections import defaultdict
import opendatasets as od
import numpy as np
from sklearn.impute import SimpleImputer
from dataloader.dataloader import donwload_data, merge_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETL:
    
    def __init__(self, map_pipes: Dict[str, List], filter_pipes: Dict[str, List]):
        self.map_pipes = map_pipes
        self.filter_pipes = filter_pipes
        

    def apply_map(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return self._apply_pipes(tables, self.map_pipes)


    def apply_filter(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return self._apply_pipes(tables, self.filter_pipes)
    

    def _apply_pipes(self, tables: Dict[str, pd.DataFrame], pipes) -> Dict[str, pd.DataFrame]:
        for table in tables:
            if table in pipes:
                pipe = pipes[table]
                tables[table] = self._apply_pipes(tables[table], pipe)
        return tables
    
    def _apply_pipe(self, tables: Dict[str, pd.DataFrame], pipe) -> pd.DataFrame:
        for func in pipe:
            table = func(table)
        return table
    

    def apply_merge(self, tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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



    def extract(self, url:str) -> Dict[str, pd.DataFrame]:

        """Downloads data from the provided url"""

        return donwload_data()
        


        


        

        
    




        


        



    




    

