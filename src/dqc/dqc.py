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
    def _check_missing_values(self, df) -> pd.DataFrame | None:

        """Calculates a percentage of missing values in each column"""

        missing_data = defaultdict()
        for col in df.columns:
            missing_data[col] = df[col].isna().mean() * 100

        logger.info(f"Missing values detection completed successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=missing_data, columns=["missing_values_percentage"], orient="index")


    @exec_time
    def _check_unique_values(self, df):

        """Calculates unique values in each column"""

        uniques = defaultdict()
        for col in df.columns:
            uniques[col] = df[col].nunique()
        
        logger.info(f"Unique values detected successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=uniques, orient="index", columns=["unique_values"])
    

    @exec_time
    def _check_outliers(self, df:pd.DataFrame) -> pd.DataFrame | None:

        """Provides statistical information on potential outliers. Calculates Q1, Q3, IQR,
        lower bound, upper bound, percentage of outliers based on IQR, min, max, 1% and 99%"""

        data = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if  numeric_cols.empty:
            logger.info(f'No numeric columns provided - {dt.datetime.now()}')
            return None

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mapping = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_percentage' : len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]) / len(df) * 100,
                'min_val': df[col].min(),
                'max_val': df[col].max(),
                '1%_percentile' : df[col].quantile(0.01),
                '99%_percentile': df[col].quantile(.99)
                }
            data.append(outlier_mapping)

        outliers_report = pd.DataFrame(data=data, index=numeric_cols)
        return outliers_report

        
    def _check_datatypes(self, df:pd.DataFrame):
        analysis = []

        for col in df.columns:
            col_data = df[col]
            current_dtype = col_data.dtype

            unique_types = set(type(x) for x in col_data.dropna())
            mixed_types = len(unique_types) > 1

            numeric_as_str = False
            if current_dtype == object:
                try:
                    pd.to_numeric(col_data.dropna())
                    numeric_as_str = True
                except:
                    pass

            possible_datetime = False
            if current_dtype == object:
                try:
                    pd.to_datetime(col_data.dropna(), errors='raise')
                    possible_datetime = True
                except:
                    pass

            suggested_dtype = current_dtype
            if numeric_as_str:
                numeric_values = pd.to_numeric(col_data, errors='coerce')
                if (numeric_values.dropna() % 1 == 0).all():
                    suggested_dtype = 'int'
                else:
                    suggested_dtype = 'float'
            elif possible_datetime:
                suggested_dtype = 'datetime'

            analysis.append({
                "column": col,
                "current_dtype": current_dtype,
                "mixed_types": mixed_types,
                "numeric_as_string": numeric_as_str,
                "possible_datetime": possible_datetime,
                "suggested_dtype": suggested_dtype
                    })
            
        report = pd.DataFrame(analysis).set_index("column")
        return report


    @exec_time
    def _check_duplicates(self, df) -> pd.DataFrame:

        """Detects fully duplicated rows"""

        duplicated = {
            "duplicates": df.duplicated().sum()
        }

        logger.info(f"Duplicates detectes successfully - {dt.datetime.now()}")
        return pd.DataFrame.from_dict(data=duplicated, orient="index", columns=["duplicated_rows"])


    @exec_time
    def _check_statistics(self, df, percentiles:List[float] = [0.01, 0.25, 0.75, 0.99]) -> pd.DataFrame | None:

        """Calculates statistics"""

        logger.info(f"Statistics calculated successfully - {dt.datetime.now()}")
        return df.select_dtypes(include=[np.number]).describe(percentiles)


    @exec_time
    def check_relationships(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        results = []
        table_names = list(tables.keys())

        for i in range(len(table_names)):
            for j in range(i + 1, len(table_names)):
                t1_name, t2_name = table_names[i], table_names[j]
                t1, t2 = tables[t1_name], tables[t2_name]
                
                common_cols = set(t1.columns).intersection(set(t2.columns))
                for col in common_cols:
                    left_ids = set(t1[col].dropna().unique())
                    right_ids = set(t2[col].dropna().unique())

                    intersect = left_ids & right_ids
                    left_only = left_ids - right_ids
                    right_only = right_ids - left_ids

                    results.append({
                        "relation": f"{t1_name}, {t2_name} ON {col}",
                        "left_only": len(left_only),
                        "intersect": len(intersect),
                        "right_only": len(right_only)
                    })

        df_results = pd.DataFrame(results)
        return df_results


    @exec_time
    def render_report(self, df) -> Dict[str, pd.DataFrame]:
        report = {
            "missing_values": self._detect_missing_values(df),
            "unique_values": self._detect_unique_values(df),
            "outliers": self._detect_outliers(df),
            "datatypes": self._check_datatypes(df),
            "duplicates": self._detect_duplicates(df),
            "statistics": self._get_statistics(df),
            "inconsistancies": self._detect_inconsistancies(df),
        }
        logger.info(f"Report generated successfully - {dt.datetime.now()}")
        return report
        
        



        


            


        


