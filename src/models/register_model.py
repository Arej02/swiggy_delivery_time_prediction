import mlflow
import dagshub
import logging
from pathlib import Path
import json
from mlflow import MlflowClient
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger=logging.getLogger('register_model')

dagshub.init(repo_owner='Arej02', repo_name='swiggy_delivery_time_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Arej02/swiggy_delivery_time_prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info=json.load(f)

    return run_info


if __name__=="__main__":
    root_path=Path(__file__).parent.parent.parent
    run_path_info=root_path/"run_information.json"
    
    run_info=load_model_information(run_path_info)

    # Get the runid and model info:
    run_id=run_info["run_id"]
    model_name=run_info["model_name"]

    # Model to register:
    model_registry_path=f"runs:/{run_id}/model"

    # Register the model:
    model_version=mlflow.register_model(
        model_uri=model_registry_path,
        name=model_name
    )

    # Get the model version:
    registeres_model_version=model_version.version
    registeres_model_name=model_version.name
    logger.info(f"The latest version in model registry is {registeres_model_version}")

    # Update the stage of the model to staging:
    client=MlflowClient()
    client.transition_model_version_stage(
        name=registeres_model_name,
        version=registeres_model_version,
        stage="Staging"
    )
    logger.info("Model is staged")

   


