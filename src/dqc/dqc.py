import pandas as pd
from collections import defaultdict
import logging 
import datetime as dt
from decorators.exec_time import exec_time
from .dqc_template import DQCTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQCPipeline(DQCTemplate):


    def validation_report(self, datasets):
            
        validation_report = defaultdict()

        validation_report["values"] = self._values_report(datasets)
        logger.info(f"Values report is ready - {dt.datetime.now()}")

        validation_report["duplicates"] = self._duplicates_report(datasets)
        logger.info(f"Duplicates report is ready - {dt.datetime.now()}")

        return validation_report


    def statistics_report(self, datasets):
            
        statistics_report = defaultdict()

        statistics_report["distribution"] = self._distribution_report(datasets)
        logger.info(f"Distribution report is ready - {dt.datetime.now()}")

        statistics_report["outliers"] = self._outliers_report(
            statistics_report["distribution"]
        )        
        logger.info(f"Outliers report is ready - {dt.datetime.now()}")

        return statistics_report
        

    def render_report(self, validation_report, statistics_report):
        rendered_report = {
            "validation_report": validation_report,
            "statistics_report": statistics_report
        }
        return rendered_report
        
        
    @exec_time
    def _values_report(self, datasets):
        report = []
        for name, dataset in datasets.items():
            nans_report = dataset.isna().sum()
            uniques_report = dataset.nunique()
            dtypes_report = dataset.infer_objects().dtypes
            totals = [len(dataset) for _ in range(len(dataset.columns))]


            report.append(pd.DataFrame(
                data = [
                    nans_report.values,
                    uniques_report.values,
                    dtypes_report,
                    totals
                ],
                index = ["nans", "uniques", "dtypes", "totals"],
                columns = pd.MultiIndex.from_tuples([(name, col) for col in dataset.columns],
                                                    names = ["table", "column"])
            ))

        return (pd.concat(report, axis=1).
                transpose().
                reset_index().
                sort_values(by="table").
                set_index(["table", "column"]))
        
    
    @exec_time
    def _duplicates_report(self, datasets):
        duplicated_tables = []
        for key in sorted(datasets.keys()):
            duplicated_tables.append(datasets[key].duplicated().sum())
        report = pd.DataFrame(data=duplicated_tables,index=sorted(datasets.keys()), columns=["duplicates_total"])
        return report
    

    @exec_time
    def _distribution_report(self, datasets):
        dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
        quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
        quantiles_tables = []
        for dataset in datasets:
            data = datasets[dataset].select_dtypes(include=dtypes)
            if not data.empty:
                quantiles_tables.append(data.describe(quantiles))
        return pd.concat(quantiles_tables, axis=1).transpose()
    

    @exec_time
    def _outliers_report(self, distribution_report):
        l_range = distribution_report["1%"] - distribution_report["min"]
        mid_range = distribution_report["99%"] - distribution_report["1%"]
        r_range = distribution_report["max"] - distribution_report["99%"]    

        columns = ["1% l_range", "98% mid_range", "1% r_range"]
        res = pd.concat([l_range, mid_range, r_range], axis=1)
        res.columns = columns
        return res

        


            


        


