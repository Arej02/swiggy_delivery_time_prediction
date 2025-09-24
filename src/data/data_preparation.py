import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger=logging.getLogger('data_preparation')


#Fetch the data:
def load_data(url:Path)->pd.DataFrame:
    try:
        df=pd.read_csv(url)
        logger.info(f"Successfully loaded data from {url}")
        return df
    except Exception as e:
        logger.exception("File not found")

#Split the dataset:
test_size=yaml.safe_load(open('params.yaml','r'))['data_preparation']['test_size']
random_state=yaml.safe_load(open('params.yaml','r'))['data_preparation']['random_state']

def split_data(df:pd.DataFrame,test_size:float,random_state:int):
    train_data,test_data=train_test_split(df,test_size=test_size,random_state=random_state)
    logger.info("Data split into train and test")
    return train_data,test_data

#Store the data
def save_data(df:pd.DataFrame,save_path:Path)->None:
    try:
        df.to_csv(save_path,index=False)
        logger.info(f"Data saved to {save_data}")
    except Exception as e:
        logger.exception("Failed to save the data")

#Path:
if __name__=="__main__":
    root_path=Path(__file__).parent.parent.parent
    data_path=root_path/"data"/"cleaned"/"swiggy_cleaned.csv"
    save_data_dir=root_path/"data"/"interim"
    save_data_dir.mkdir(exist_ok=True,parents=True)
    train_filename="train.csv"
    test_filename="test.csv"
    save_train_path=save_data_dir/train_filename
    save_test_path=save_data_dir/test_filename

    df=load_data(data_path)
    train_data,test_data=split_data(df,test_size=test_size,random_state=random_state)
    logger.info("Parameters read")

    # save train data
    save_data(train_data,save_train_path)

    # save test data
    save_data(test_data,save_test_path)




