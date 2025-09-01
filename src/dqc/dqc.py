import pandas as pd
from collections import defaultdict
import logging 
import datetime as dt
from decorators.exec_time import exec_time
from typing import Optional, Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQCPipeline:
        

    def validation_report(self, tables:Dict[str, pd.DataFrame]):
        validation_report = {}
        validation_report['missings'] = self._check_missing_values(tables)
        validation_report['uniques'] = self._check_unique_values(tables)
        validation_report['datatypes'] = self._check_datatypes(tables)
        validation_report['relationships'] = self._check_relationships(tables)

        return validation_report
    

    def statistics_report(self, tables:Dict[str, pd.DataFrame]):
        statistics_report = {}
        statistics_report['outliers'] = self._check_outliers(tables)
        statistics_report['statistics'] = self._check_statistics(tables)

        return statistics_report
    

    def render_report(self, val_rep, stats_rep):
        report = {
            'validation_report' : val_rep,
            'statistics_report' : stats_rep
        }
        return report

        
     
    @exec_time
    def _check_missing_values(self, tables:Dict[str, pd.DataFrame]):
        report = []
        for t_name, df in tables.items():
            missings = df.isna().sum(0) / len(df)
            report.append(pd.DataFrame(
                data=[missings.values],
                index=["missings_percentage"],
                columns=pd.MultiIndex.from_tuples([(t_name, col) for col in df.columns],names=["table_name", "column_name"])))

        return (pd.concat(report, axis=1).
                transpose().
                reset_index().
                sort_values('table_name').
                set_index(["table_name", "column_name"])
                )


    @exec_time
    def check_unique_values(self, tables:Dict[str, pd.DataFrame]):
        report = []
        for t_name, df in tables.items():
            uniques = df.nunique()

            report.append(pd.DataFrame(
                data=[
                    uniques.values
                ],
                index=["unique_values"],
                columns=pd.MultiIndex.from_tuples([(t_name, col) for col in df.columns],
                                                  names=[
                                                  "table_name", "column_name"])
            ))

        return (pd.concat(report, axis=1).
                transpose().
                reset_index().
                sort_values('table_name').
                set_index(["table_name", "column_name"])
                )
    

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


    def _suggest_dtype(series: pd.Series):
        if pd.api.types.is_integer_dtype(series):
            return pd.to_numeric(series, downcast="integer").dtype
        elif pd.api.types.is_float_dtype(series):
            return pd.to_numeric(series, downcast="float").dtype
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        elif pd.api.types.is_object_dtype(series):
            if series.nunique() / len(series) < 0.5:
                return "category"
            else:
                return "object"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime64[ns]"
        else:
            return series.dtype

    def _check_datatypes(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        report = []

        for t_name, df in tables.items():
            current_dtypes = df.dtypes.astype(str).values
            infer_dtypes = df.infer_objects().dtypes.astype(str).values

            is_mixed = current_dtypes != infer_dtypes

            suggested_dtypes = [str(self._suggest_dtype(df[col])) for col in df.columns]

            report.append(
                pd.DataFrame(
                    data=[current_dtypes, infer_dtypes, is_mixed, suggested_dtypes],
                    index=["current_dtype", "inferred_dtype", "is_mixed", "suggested_dtype"],
                    columns=pd.MultiIndex.from_tuples(
                        [(t_name, col) for col in df.columns],
                        names=["table_name", "column_name"]
                    )
                )
            )

        return (
            pd.concat(report, axis=1)
            .transpose()
            .reset_index()
            .sort_values("table_name")
            .set_index(["table_name", "column_name"])
        )



    @exec_time
    def check_duplicates(tables:Dict[str, pd.DataFrame]) -> pd.DataFrame:

        duplicates = {}
        for table, df in tables.items():
            duplicates[table] = df.duplicated().sum()           
        return pd.DataFrame.from_dict(data=duplicates, orient="index", columns=["duplicated_rows"])


    @exec_time
    def check_statistics(tables: Dict[str, pd.DataFrame], percentiles: List[float] = [0.01, 0.25, 0.75, 0.99]) -> pd.DataFrame:
        report = []

        for table_name, df in tables.items():
            desc = df.describe(percentiles=percentiles).transpose()
            desc["table_name"] = table_name
            desc["column_name"] = desc.index
            report.append(desc)

        result = pd.concat(report, ignore_index=True)
        return result.set_index(["table_name", "column_name"])


    @exec_time
    def _check_relationships(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:

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
        
        



        


            


        


