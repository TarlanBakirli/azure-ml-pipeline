import os
from pathlib import Path
from mldesigner import command_component, Input, Output
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

@command_component(
    name="stock_predictor_prep_data",
    version="1",
    display_name="Generate and Preprocess Synthetic Data",
    description="Generate synthetic timeseries stock prices data and preprocess the data",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
)
def prepare_data_component(
    training_data: Output(type="uri_folder"),
    previous_time_window: int,
    next_time_window: int,
    test_size: float,
):
    
    # TODO: Extract execution logic into separate .py file

    # Generate synthetic data
    sequence = generate_stock_prices()

    # Normalize data
    normalized_sequence = normalize(sequence)

    # Split sequence into chunks
    X, y = split_sequence(normalized_sequence, previous_time_window, next_time_window)

    # Split into train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size)

    np.save(os.path.join(training_data, "X_train.npy"), X_train)
    np.save(os.path.join(training_data, "y_train.npy"), y_train)


def generate_stock_prices(n_points=500, initial_price=100, volatility=0.02):
    # Generate random returns using normal distribution
    returns = np.random.normal(loc=0.0001, scale=volatility, size=n_points)
    
    # Calculate price path using cumulative returns
    price_path = initial_price * np.exp(np.cumsum(returns))
    
    return price_path

def normalize(sequence):
    # Normalize data
    sequence = sequence.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(sequence)
    normalized_sequence = scaler.transform(sequence)
    normalized_sequence = normalized_sequence.flatten()
    return normalized_sequence

def split_sequence(normalized_sequence, previous_time_window, next_time_window):
    # Split normalized sequence into chunks
    X, y = [], []
    for i in range(len(normalized_sequence) - previous_time_window - next_time_window + 1):
        X.append(normalized_sequence[i:(i + previous_time_window)])
        y.append(normalized_sequence[(i + previous_time_window):(i + previous_time_window + next_time_window)])
    
    return X, y

def split_train_test(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=1)