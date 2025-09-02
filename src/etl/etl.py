import pandas as pd
import os
import sys
import logging
from decorators.exec_time import exec_time
import datetime as dt
from typing import Dict, List, Tuple, Callable, Any
from collections import defaultdict
import opendatasets as od


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETL:

    def extract(self, url:str) -> Dict[str, pd.DataFrame]:

        od.download(url)
        dataset_name = url.split("/")[-1]
        dataframes = defaultdict()

        for dirname, _, filenames in os.walk(f'{os.getcwd()}/{dataset_name}'):
            for filename in filenames:
                dataframes[filename.split(".")[0]] = pd.read_csv(f"{dirname}/{filename}")

        logger.info(f'Data extracted succesfully - {dt.datetime.now()}')
        return dataframes
    

    def _apply_pipes(self, tables: Dict[str, pd.DataFrame], pipes: Dict[str, Callable[..., Any]]) -> Dict[str, pd.DataFrame]:
        for table in tables:
            if table in pipes:
                pipe = pipes[table]
                tables[table] = self._apply_pipe(tables[table], pipe)
        logger.info(f'Pipes aplied sucessfully - {dt.datetime.now()}')
        return tables
    

    def _apply_pipe(self, table: pd.DataFrame, pipe: Callable[..., Any]) -> pd.DataFrame:
        for func in pipe:
            table = func(table)
        return table
    

    def _transform(self, tables: Dict[str, pd.DataFrame], pipes: Dict[str, Callable[..., Any]]) -> Dict[str, pd.DataFrame]:
        
        logger.info(f'Data transformed successfully - {dt.datetime.now()}')
        return self._apply_pipes(tables, pipes)


    def _apply_merge(self, tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        logger.info(f'Tables merged successfully - {dt.datetime.now()}')
        return merged_tables_train, merged_tables_test

    @exec_time
    def load(self, url:str, pipes: Dict[str, Callable[..., Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.extract(url)
        data = self._transform(tables=data, pipes=pipes)
        data = self._apply_merge(tables = data)
        logger.info(f'Data loaded successfully - {dt.datetime.now()}')
        return data
        


        


        

        
    




        


        



    




    

