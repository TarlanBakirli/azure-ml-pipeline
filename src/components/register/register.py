from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from pathlib import Path

def get_file(f):

    f = Path(f)
    if f.is_file():
        if f.name == "pipeline_stock_predictor_model.pt":
            return f
        else:
            raise RuntimeError("Provided input file is not torch model")
    else:
        for file in list(f.iterdir()):
            if file.name == "pipeline_stock_predictor_model.pt":
                return file # returns full path to the model
    

def register(model_input_file, registered_model_name, ml_client):

    model_file_path = get_file(model_input_file)

    cloud_model = Model(
        path=model_file_path,
        name=registered_model_name,
        type=AssetTypes.CUSTOM_MODEL,
        description="Pipeline Stock Predictor Model",
    )
    ml_client.models.create_or_update(cloud_model)