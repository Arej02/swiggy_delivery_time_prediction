import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import set_config

set_config(transform_output="pandas")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger=logging.getLogger('data_preprocessing')

target_col = "time_taken"

#Load the data:
def load_data(url:Path)->pd.DataFrame:
    try:
        df=pd.read_csv(url)
        logger.info(f"Read the csv file from {url}")
        return df
    except FileNotFoundError as e:
        logger.info("File not found")

def drop_missing_values(df:pd.DataFrame)->pd.DataFrame:
    try:
        missing_values=df.isnull().sum().sum()
        logger.info(f"There are {missing_values} rows")
        rows_before=df.shape[0]
        df_dropped=df.dropna()
        rows_after=df_dropped.shape[0]
        logger.info(f"{rows_before-rows_after} rows dropped")
        return df_dropped
    except Exception as e:
        logger.info("Some error occured while dropping values")

def train_preprocessor(df:pd.DataFrame,preprocessor):
    preprocessor.fit(df)
    logger.info(f"Data fitted by {preprocessor}")
    return preprocessor

def perform_transformation(df:pd.DataFrame,preprocessor)->pd.DataFrame:
    transformed_data=preprocessor.transform(df)
    logger.info(f"Data transformed by {preprocessor}")
    return transformed_data

def split_data(df:pd.DataFrame,target_col:str):
    X=df.drop(columns=[target_col])
    Y=df[target_col]
    logger.info("Your data has been splitted")
    return X,Y

def join_data(df:pd.DataFrame,y:pd.Series)->pd.DataFrame:
    joined_df=pd.concat([df,y],axis=1)
    logger.info("Features and target joined")
    return joined_df

def save_data(df:pd.DataFrame,url:Path)->None:
    try:
        df.to_csv(url,index=False)
        logger.info(f"Saved to {url}")
    except Exception as e:
        logger.info("The file could not be saved")

def save_transformer(transformer,save_dir:Path,name:str):
    save_path=save_dir/f"{name}.joblib"
    joblib.dump(transformer,save_path)
    logger.info(f"Transformer saved to the path {save_path}")


num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]



# generate order for ordinal encoding

traffic_order = ["low","medium","high","jam"]

distance_type_order = ["short","medium","long","very_long"]


if __name__=="__main__":
    root_path=Path(__file__).parent.parent.parent
    save_path=root_path/"data"/"processed"
    save_path.mkdir(exist_ok=True,parents=True)
    train_data_path=root_path/"data"/"interim"/"train.csv"
    test_data_path=root_path/"data"/"interim"/"test.csv"

    train_trans_filename="train_trans.csv"
    test_trans_filename="test_trans.csv"

    save_path_train=save_path/train_trans_filename
    save_path_test=save_path/test_trans_filename

    transformer_dir=root_path/"models"
    transformer_dir.mkdir(exist_ok=True, parents=True)

    preprocessor = ColumnTransformer(transformers=[
            ("scale", StandardScaler(), num_cols),
            ("nominal_encode", OneHotEncoder(drop="first",
                                            handle_unknown="ignore",
                                            sparse_output=False), nominal_cat_cols),
            ("ordinal_encode", OrdinalEncoder(categories=[traffic_order,
                                                          distance_type_order],
                                            encoded_missing_value=-999,
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1), ordinal_cat_cols)],
                                    remainder="passthrough",
                                    n_jobs=-1,
                                    force_int_remainder_cols=False,
                                    verbose_feature_names_out=False)
    
    # Load the test and train data and drop the missing values:
    train_df=load_data(train_data_path)
    train_df=drop_missing_values(train_df)

    test_df=load_data(test_data_path)
    test_df=drop_missing_values(test_df)

    # Split the dataset:
    X_train,Y_train=split_data(train_df,target_col)
    X_test,Y_test=split_data(test_df,target_col)

    # Fit and transform the data:
    train_preprocessor(X_train,preprocessor)
    X_train_trans=perform_transformation(X_train,preprocessor)
    X_test_trans=perform_transformation(X_test,preprocessor)

    # join the data:
    train_data_df=join_data(X_train_trans,Y_train)
    test_data_df=join_data(X_test_trans,Y_test)

    # Save the transformed data:
    save_data(train_data_df,save_path_train)
    save_data(test_data_df,save_path_test)

    # Save transformer
    save_transformer(preprocessor,transformer_dir,"preprocessor")

