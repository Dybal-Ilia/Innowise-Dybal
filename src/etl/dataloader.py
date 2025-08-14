import os
import sys
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data():
    cwd = os.getcwd()
    src_dir = os.path.join(cwd, '..', 'src')
    sys.path.append(cwd)

    data_path = f"{src_dir}/data"
    if not os.path.exists(data_path) or not os.listdir(data_path):

        with open(f"{src_dir}/etl/kaggle.json", "r") as f:
            creds = json.load(f)

        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']

        os.system(
            f"kaggle competitions download -c competitive-data-science-predict-future-sales -p {data_path}")
        os.system(
            f'yes "yes" | unzip {data_path}/competitive-data-science-predict-future-sales -d {data_path}')
        os.system(f"rm {data_path}/competitive-data-science-predict-future-sales.zip")