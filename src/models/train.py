import pandas as pd
import numpy as np
import logging
import joblib
import yaml
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger=logging.getLogger('train_model')

target_col="time_taken"

def load_data(url:Path)->pd.DataFrame:
    try:
        df=pd.read_csv(url)
        logger.info("Dataset is read")
        return df
    except FileNotFoundError as e:
        logger.info("File doesnot exist")


lightgbm_params=yaml.safe_load(open('params.yaml','r'))['models']['lightgbm']

rf_params=yaml.safe_load(open('params.yaml','r'))['models']['random_forest']

def train_model(model,X_train:pd.DataFrame,y_train):
    model.fit(X_train,y_train)
    logger.info(f"{model} has been trained")
    return model

def split_data(df:pd.DataFrame,target_col):
    X=df.drop(columns=[target_col])
    logger.info("Features to be trained created")
    Y=df[target_col]
    logger.info("Target column made")
    return X,Y

def save_model(model,url:Path,model_name:str):
    save_location=url/model_name
    joblib.dump(value=model,filename=save_location)
    logger.info(f"{model} has been saved")

def save_transformer(transformer,url:Path,transformer_name:str):
    save_location=url/transformer_name
    joblib.dump(value=transformer,filename=save_location)
    logger.info(f"{transformer} has been saved")


if __name__=="__main__":

    root_path=Path(__file__).parent.parent.parent
    data_path=root_path/"data"/"processed"/"train_trans.csv"

    # Load the training data:
    traning_data=load_data(data_path)

    # Split the data into X and Y:
    X_train,Y_train=split_data(traning_data,target_col)

    # Build Model:
    rf=RandomForestRegressor(**rf_params)
    logger.info("Random Forest Regressor made")

    lgbm=LGBMRegressor(**lightgbm_params)
    logger.info("LightGBM model made")

    lr=LinearRegression()
    logger.info("Linear Regression model made")

    # Power Transformer:
    pf=PowerTransformer()
    logger.info("Power Transformer made")

    # Stacking Regressor:
    stacking_reg = StackingRegressor(estimators=[("rf_model",rf),
                                                 ("lgbm_model",lgbm)],
                                     final_estimator=lr,
                                     cv=3,n_jobs=-1)
    logger.info("Stacking regressor built")

    # Make the model wrapper:
    model = TransformedTargetRegressor(regressor=stacking_reg,
                                       transformer=pf)
    logger.info("Models wrapped inside wrapper")
    
    # Train the model:
    train_model(model,X_train,Y_train)

    # Model name:
    model_filename="model.joblib"
    stacking_filename="stacking_regressor.joblib"
    transformer_filename = "power_transformer.joblib"
    model_sav_dir=root_path/"models"
    model_sav_dir.mkdir(exist_ok=True)

    stacking_model=model.regressor_
    transformer=model.transformer_

    # Save model
    save_model(model,model_sav_dir,model_filename)
    save_model(stacking_model,model_sav_dir,stacking_filename)
    save_transformer(transformer,model_sav_dir,transformer_filename)