import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from argparse import ArgumentParser
import pandas as pd
import logging
import os
from sklearn.pipeline import Pipeline

logging.basicConfig(filename="logs\\process_exec.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess(X):
    """
    Preprocess raw training data.

    - Loads CSV from given path
    - Applies feature engineering and transformations
    - Saves processed dataset to data/pre_processed folder
    """

    # dropping columns not needed for training
    cols_to_drop = ['PassengerId', 'Name']
    if 'Transported' in X.columns.to_list():
        cols_to_drop.append('Transported')
    X_processed = X.copy()
    X_processed = X_processed.drop(columns=cols_to_drop)

    logging.info(f"columns dropped successfully: {cols_to_drop}")

    # extracting numerical columns to process them
    num_cols = X_processed.select_dtypes(include=['float64']).columns.to_list()

    # applying log1p since these columns were found to be skewed in the EDA
    for i in num_cols:
        X_processed[i] = np.log1p(X_processed[i])

    logging.info(f"transformation log1p applied to numerical columns ({num_cols})")

    # convert Cabin to 3 new cols, and drop the original Cabin col
    Cabin_list = X_processed.Cabin.apply(lambda x: x.split('/') if pd.notnull(x) and x.count('/') == 2 else [None,None,None])

    X_processed['CabinDeck'] = Cabin_list.apply(lambda x: x[0])
    X_processed['CabinNum'] = Cabin_list.apply(lambda x: float(x[1]) if x[1] is not None else None)
    X_processed['CabinSide'] = Cabin_list.apply(lambda x: x[2])

    X_processed = X_processed.drop('Cabin', axis=1)

    logging.info(f"column 'Cabin' split into 3 columns (['CabinDeck','CabinNum','CabinSide'])")

    # redefining these variables based on changes made so far
    num_cols = X_processed.select_dtypes(include=['number']).columns.to_list()
    cat_cols = X_processed.select_dtypes(include=['object','string']).columns.to_list()

    # creating a column transformer - to handle the following
    # Null value imputation for numerical cols 
    # One-hot encoding for categorical cols

    preprocessor = ColumnTransformer(
            transformers=[
            ("num", SimpleImputer(strategy='median'), num_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
                ],
            remainder="passthrough"
        )

    # applying the preprocessor on X_processed
    X_processed = preprocessor.fit_transform(X_processed)
    new_feature_names = preprocessor.get_feature_names_out()

    logging.info(f"null value imputation, and one hot encoding applied to respective columns")

    return pd.DataFrame(X_processed,
                columns=new_feature_names,
                index=X.index)

if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--raw_file_name",type=str,default="train.csv")

    args = arg_parser.parse_args()

    logging.info("starting preprocessing script")

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

    X_processed = preprocess(X)

    try:
        output_dir = "data/pre_processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            args.raw_file_name
        )
        X_processed.to_csv(output_path,index=False)
    except Exception as e:
            logging.error(f"unexpected error, while saving final csv: {e}")

    logging.info(f"prerocessed file saved to disk at this path: {f"data\\pre_processed\\{path_to_raw_data.split('\\')[-1].split(".")[0]}_processed.csv"}")

    logging.info(f"ending preprocessing script")
    