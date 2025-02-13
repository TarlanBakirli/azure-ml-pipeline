from pathlib import Path
from mldesigner import command_component, Input, Output
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

@command_component(
    name="stock_predictor_register_model",
    version="1",
    display_name="Register trained model in the workspace",
    description="Register trained model in the workspace",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def register_component(
    input_data: Input(type="uri_folder"),
    registered_model_name: str,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
):
    from register import register


    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name
    )

    register(input_data, registered_model_name, ml_client)