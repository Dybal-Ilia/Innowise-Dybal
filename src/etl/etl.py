import pandas as pd
import os
import sys
import logging
from decorators.exec_time import exec_time
import datetime as dt
from typing import Dict, List, Tuple, Callable, Any
import opendatasets as od
from dataloader.dataloader import donwload_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETL:
    

    def _apply_pipes(self, tables: Dict[str, pd.DataFrame], pipes: Dict[str, Callable[..., Any]]) -> Dict[str, pd.DataFrame]:
        for table in tables:
            if table in pipes:
                pipe = pipes[table]
                tables[table] = self._apply_pipe(tables[table], pipe)
        return tables
    

    def _apply_pipe(self, table: Dict[str, pd.DataFrame], pipe: Callable[..., Any]) -> pd.DataFrame:
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
        


        


        

        
    




        


        



    




    

