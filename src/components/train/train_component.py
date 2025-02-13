from pathlib import Path
from mldesigner import command_component, Input, Output


@command_component(
    name="stock_predictor_train",
    version="1",
    display_name="Train Simple Stock Predictor using PyTorch",
    description="Train Simple Stock Predictor using PyTorch",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def train_component(
    input_data: Input(type="uri_folder"),
    output_model: Output(type="uri_folder"),
    epochs=1,
    learning_rate=0.01,
    previous_time_window=10,
    next_time_window=5,
):
    from train import train

    train(input_data, output_model, epochs, learning_rate, previous_time_window, next_time_window)