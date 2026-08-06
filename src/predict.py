from datetime import datetime
import os
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from argparse import ArgumentParser
from preprocess import preprocess
import logging

mlflow.set_tracking_uri('sqlite:///mlflow.db')
logging.basicConfig(filename="logs\\process_exec.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def predict(X_raw,m_type="latest"):
    """Gets the current model marked as 'Champion', i.e., best model
    and predicts labels"""

    logging.info("preprocessing the raw dataframe to get X_processed")
    X_processed = preprocess(X_raw)

    if m_type == 'champion':
            logging.info("loading 'Champion' model from mlflow registry")
            loaded_model = mlflow.sklearn.load_model("models:/Champion/latest")
    else:
        logging.info("loading 'latest' model from mlflow runs")
        latest_run = mlflow.search_runs(
            experiment_ids=[1],
            filter_string="metric.accuracy_score > 0",
            order_by=["start_time DESC"],   # latest run first
            max_results=1
        )
        latest_run_id = latest_run.run_id[0]
        loaded_model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")

    logging.info(f"using '{m_type}' model to make predictions")
    y_prob = loaded_model.predict_proba(X_processed)
    current_threshold = 0.5584083119472486
    y_pred = y_prob[:,1] > current_threshold

    logging.info("using y_pred to create an output dataframe")
    result_df = pd.DataFrame({
        "PassengerId": X_raw["PassengerId"],
        "Transported": y_pred
    })

    return result_df


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--raw_file_name",type=str,default="test.csv")
    arg_parser.add_argument("--m_type",type=str,default="latest")

    args = arg_parser.parse_args()

    logging.info("starting predict script")

    input_dir = "data/raw/"
    os.makedirs(input_dir, exist_ok=True)
    path_to_raw_data = os.path.join(input_dir,args.raw_file_name)

    # initializing the raw dataframe
    try:
        X = pd.read_csv(path_to_raw_data)
    except FileNotFoundError:
        logging.error(f"file not found at this path: {path_to_raw_data}")
    except Exception as e:
        logging.error(f"unexpected error, while reading csv: {e}")

    logging.info(f"csv file loaded successfully from path: {path_to_raw_data}")

    logging.info(f"calling evaluate function with this raw csv file as input df")
    result_df = predict(X,args.m_type)

    # Get current date and time, and Format as yyyymmdd_hhmmss
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    output_dir = "data/output/"
    os.makedirs(output_dir, exist_ok=True)
    path_to_output_csv = os.path.join(output_dir,f"{args.raw_file_name.split(".")[0]}_{timestamp_str}.csv")

    logging.info(f"coverting result_df into a csv and storing at {path_to_output_csv}")
    result_df.to_csv(path_to_output_csv, index=False, encoding='utf-8')