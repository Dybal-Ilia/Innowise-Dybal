import pandas as pd
from collections import defaultdict
import logging 
import datetime as dt
from decorators.exec_time import exec_time
from typing import Optional, Dict
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQCPipeline:
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        self.df = df
        self.config = config if config else {}
        self.defaultconfig = {
            "percentiles": [0.01, 0.99]
        }
        self.config = {**self.defaultconfig, **self.config}
       
    
    @exec_time
    def _detect_missing_values(self) -> pd.DataFrame | None:

        """Calculates a percentage of missing values in each column"""

        missing_data = defaultdict()
        for col in self.df.columns:
            missing_data[col] = self.df[col].isna().mean() * 100
        return pd.DataFrame.from_dict(data=missing_data, columns=["missing_values_percentage"], orient="index")


    @exec_time
    def _detect_unique_values(self):

        """Calculates unique values in each column"""

        uniques = defaultdict()
        for col in self.df.columns:
            uniques[col] = self.df[col].nunique()
        return pd.DataFrame.from_dict(data=uniques, orient="index", columns=["unique_values"])
    

    @exec_time
    def _detect_outliers(self) -> pd.DataFrame | None:

        """Calculates the amount of outliers in a dataframe based on Isolation Forest"""
        X = self.df.select_dtypes(include=[np.number])
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, verbose = 1)
        preds = iso.fit_predict(X)
        outliers = {
            "outliers" : len(preds[preds == -1])
        }
        return pd.DataFrame.from_dict(data=outliers, orient="index", columns=["outliers_total"])

        
    @exec_time
    def _detect_duplicates(self) -> pd.DataFrame | None:

        """Detects fully duplicated rows"""

        duplicated = {
            "duplicates": self.df.duplicated().sum()
        }
        return pd.DataFrame.from_dict(data=duplicated, orient="index", columns=["duplicated_rows"])


    @exec_time
    def _get_statistics(self) -> pd.DataFrame | None:

        """Calculates statistics"""

        return self.df.select_dtypes(include=[np.number]).describe(self.config["percentiles"])


    @exec_time 
    def _detect_inconsistencies(self) -> pd.DataFrame | None:

        """Detects negative values"""

        negatives = defaultdict()
        for col in self.df.select_dtypes(include=[np.number]).columns:
            negatives[col] = len(self.df[self.df[col] < 0])    
        return pd.DataFrame.from_dict(data=negatives, orient="index", columns=["negative_values"]) 
    

    @exec_time
    def render_report(self) -> Dict[str, pd.DataFrame]:
        report = {
            "missing_values": self._detect_missing_values(),
            "unique_values": self._detect_unique_values(),
            "outliers": self._detect_outliers(),
            "duplicates": self._detect_duplicates(),
            "statistics": self._get_statistics(),
            "inconsistencies": self._detect_inconsistencies()
        }
        return report
        
        



        


            


        


