import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json


dagshub.init(repo_owner='Arej02', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Arej02/swiggy_delivery_time_prediction.mlflow")
mlflow.set_experiment("DVC pipeline")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger=logging.getLogger('model_evaluation')

target_col="time_taken"


def load_data(url:Path)->pd.DataFrame:
    try:
        df=pd.read_csv(url)
        logger.info("Dataset is read")
        return df
    except FileNotFoundError as e:
        logger.info("File doesnot exist")

def split_data(df:pd.DataFrame,target_col):
    X=df.drop(columns=[target_col])
    logger.info("Features to be trained created")
    Y=df[target_col]
    logger.info("Target column made")
    return X,Y

def load_model(url:Path):
    model=joblib.load(url)
    logger.info("Model has been read")
    return model

if __name__=="__main__":

    root_path=Path(__file__).parent.parent.parent
    train_data_path=root_path/"data"/"processed"/"train_trans.csv"
    test_data_path=root_path/"data"/"processed"/"test_trans.csv"

    model_path=root_path/"models"/"model.joblib"

    # Load the training data:
    traning_data=load_data(train_data_path)
    testing_data=load_data(test_data_path)

    # Split the data into X and Y:
    X_train,Y_train=split_data(traning_data,target_col)
    X_test,Y_test=split_data(testing_data,target_col)

    # Build Model:
    model=load_model(model_path)

    # Get the predictions:
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    logger.info("Prediciton of data completed")

    # Performance metrics:
    train_mae=mean_absolute_error(Y_train,y_train_pred)
    test_mae=mean_absolute_error(Y_test,y_test_pred)
    logger.info("Mean absolute error calculated")

    train_r2=r2_score(Y_train,y_train_pred)
    test_r2=r2_score(Y_test,y_test_pred)
    logger.info("R2 score calculated")


    cv=cross_val_score(
        model,X_train,Y_train,cv=3,scoring='neg_mean_absolute_error',n_jobs=-1
    )
    logger.info("Cross Validation calculated")
    scores=-cv.mean()

    with mlflow.start_run(run_name="Final model"):

        # Set tag
        mlflow.set_tag("model","Food Delivery Time Regressor")

        # Log parameters:
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)

        # Log metrics:
        mlflow.log_metric("training_mae",train_mae)
        mlflow.log_metric("testing_mae",test_mae)
        mlflow.log_metric("training_r2",train_r2)
        mlflow.log_metric("testing_r2",test_r2)
        mlflow.log_metric("cross validation",scores)

        # Log Dataset:
        train_data_input = mlflow.data.from_pandas(traning_data,targets=target_col)
        test_data_input = mlflow.data.from_pandas(testing_data,targets=target_col)
        
        # log input
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")

        logger.info("Mlflow logging completed")
