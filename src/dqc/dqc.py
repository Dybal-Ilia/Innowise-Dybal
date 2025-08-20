import pandas as pd
from collections import defaultdict
import logging 
import datetime as dt
from decorators.exec_time import exec_time
from typing import Optional, Dict, List
import numpy as np
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQCPipeline:
        
    
    @exec_time
    def _detect_missing_values(self, df) -> pd.DataFrame | None:

        """Calculates a percentage of missing values in each column"""

        missing_data = defaultdict()
        for col in df.columns:
            missing_data[col] = df[col].isna().mean() * 100

        logger.info(f"Missing values detection completed successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=missing_data, columns=["missing_values_percentage"], orient="index")


    @exec_time
    def _detect_unique_values(self, df):

        """Calculates unique values in each column"""

        uniques = defaultdict()
        for col in df.columns:
            uniques[col] = df[col].nunique()
        
        logger.info(f"Unique values detected successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=uniques, orient="index", columns=["unique_values"])
    

    @exec_time
    def _detect_outliers(self, df) -> pd.DataFrame | None:

        """Calculates the amount of outliers in a dataframe based on Isolation Forest"""
        X = df[["item_price", "item_cnt_day"]]
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, verbose = 1)
        preds = iso.fit_predict(X)
        outliers = {
            "outliers" : len(preds[preds == -1])
        }

        logger.info(f"Outliers detected successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=outliers, orient="index", columns=["outliers_total"])

        
    @exec_time
    def _detect_duplicates(self, df) -> pd.DataFrame | None:

        """Detects fully duplicated rows"""

        duplicated = {
            "duplicates": df.duplicated().sum()
        }

        logger.info(f"Duplicates detectes successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=duplicated, orient="index", columns=["duplicated_rows"])


    @exec_time
    def _get_statistics(self, df, percentiles:List[float] = [0.01, 0.25, 0.75, 0.99]) -> pd.DataFrame | None:

        """Calculates statistics"""

        logger.info(f"Statistics calculated successfully - {dt.datetime.now()}")
        return df.select_dtypes(include=[np.number]).describe(percentiles)


    @exec_time 
    def _detect_inconsistancies(self, df) -> pd.DataFrame | None:

        """Detects Inconsistancies"""

        negatives = defaultdict()
        for col in df.select_dtypes(include=[np.number]).columns:
            negatives[col] = len(df[df[col] < 0])    

        logger.info(f"Inconsistancies detected successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=negatives, orient="index", columns=["negative_values"]) 
    

    @exec_time
    def render_report(self, df) -> Dict[str, pd.DataFrame]:
        report = {
            "missing_values": self._detect_missing_values(df),
            "unique_values": self._detect_unique_values(df),
            "outliers": self._detect_outliers(df),
            "duplicates": self._detect_duplicates(df),
            "statistics": self._get_statistics(df),
            "inconsistancies": self._detect_inconsistancies(df)
        }
        logger.info(f"Report generated successfully - {dt.datetime.now()}")
        return report
        
        



        


            


        


