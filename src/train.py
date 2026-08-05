import logging
import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
import mlflow
from datetime import datetime
from preprocess import preprocess

mlflow.set_experiment("space-titanic-problem")
logging.basicConfig(filename="logs\\process_exec.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train(raw_X_train_with_y):
    """
    Train the model using the preprocessed data file
    """
    
    X_train_raw = raw_X_train_with_y.drop(columns=['Transported'])
    y_train_raw = raw_X_train_with_y['Transported']

    logging.info(f"preprocessing the raw train dataset")
    X_train_processed = preprocess(X_train_raw)

    logging.info(f"splitting the processed data into train and test datasets")
    X_train, X_test, y_train, y_test = train_test_split(X_train_processed, y_train_raw, test_size=0.3, random_state=11)

    now = datetime.now()
    datestr = now.strftime(f"%Y-%m-%d_%H-%M-%S")
    model_type = args.model_type
    run_name = f"train_run_{model_type}_{datestr}"

    logging.info(f"starting the mlflow train run: {run_name}")
    with mlflow.start_run(run_name=run_name):

        max_iter = 10000

        model = LogisticRegression(
            max_iter=max_iter, 
            verbose=1, 
            random_state=11
            )

        mlflow.log_param('max_iter', max_iter)

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, name="model",skops_trusted_types=['numpy.dtype'])

        logging.info(f"getting metrics for trained model")
        current_threshold = 0.5
        mlflow.log_param('current_threshold', current_threshold)
        y_prob = model.predict_proba(X_test)
        y_pred = y_prob[:, 1] >= current_threshold

        mlflow.log_metric('accuracy_score',accuracy_score(y_test, y_pred))
        logging.info(f"accuracy on test dataset: {accuracy_score(y_test, y_pred)}")
        
        conf_mattix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mattix, display_labels=model.classes_)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")

        auc_score = roc_auc_score(y_test, y_prob[:,1])
        mlflow.log_metric('auc_score',auc_score)
        logging.info(f"auc score on test dataset: {auc_score}")

        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])

        fig = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0,1], [0,1], 'k--', label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        mlflow.log_figure(fig,"roc_curve.png")

        # also getting the optimal threshold
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        optimal_threshold = thresholds[optimal_idx]
        mlflow.log_param('optimal_threshold',optimal_threshold)
        logging.info(f"optimal threshold for prediction: {optimal_threshold}")

    return 0

if __name__== "__main__":
        
        arg_parser = ArgumentParser()
        arg_parser.add_argument("--raw_file_name",type=str,default="train.csv")
        arg_parser.add_argument("--model_type",type=str,default="logistic-regression")
    
        args = arg_parser.parse_args()

        raw_data_dir = "data/raw"
        os.makedirs(raw_data_dir, exist_ok=True)
        path_to_raw_data = os.path.join(raw_data_dir,args.raw_file_name)

        logging.info(f"loading raw train data csv from path: {path_to_raw_data}")
        raw_X_train_with_y = pd.read_csv(path_to_raw_data)

        train(raw_X_train_with_y)