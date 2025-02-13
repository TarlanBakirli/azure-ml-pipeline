import torch
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import argparse

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, __version__

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
def get_workspace_handle():
    # -- Get a handle to Azure workspace --
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        credential = InteractiveBrowserCredential()

    ml_client = MLClient.from_config(credential=credential)

    return ml_client

def download_model_to_local(ml_client, model_name, local_dir, version):
    if not version:
        models = ml_client.models.list(name=model_name)
        latest_model_version = max(int(m.version) for m in list(models))
        version = latest_model_version

    os.makedirs(local_dir, exist_ok=True)

    ml_client.models.download(
        name=model_name,
        version=version,
        download_path=local_dir
    )

def get_test_data(previous_time_window, next_time_window):
    # Generate some test data
    sequence = generate_stock_prices()

    # Normalize data
    normalized_sequence = normalize(sequence)

    X, y = split_sequence(normalized_sequence, previous_time_window, next_time_window)
    X = torch.Tensor(X).float()

    # Last row in the chunk data corresponds to last previous_time_window days, thus we input that to get next_time_window days prices
    X = X[-1]

    return X

def predict(model_name, model_version, previous_time_window, next_time_window, vis):

    ml_client = get_workspace_handle()

    local_dir = f"./downloaded_models/"
    download_model_to_local(ml_client, model_name, local_dir, model_version)

    model_path = os.path.join(local_dir, model_name, "pipeline_stock_predictor_model.pt")  
    torch_model = torch.load(model_path, weights_only=False)

    X = get_test_data(previous_time_window, next_time_window)

    pred = torch_model(X)

    print(pred)

    # Visualize
    if vis:
        pred = pred.detach().numpy()
        dates = np.arange(next_time_window)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, pred, color='blue', linewidth=1)
        plt.title('Stock Prices for Next 5 days', fontsize=14, pad=15)
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        # plt.show()

        vis_dir = f"./visuals/"
        os.makedirs(vis_dir, exist_ok=True)
        image_path = os.path.join(vis_dir, f"next_{next_time_window}_days_stock_prices.png")
        plt.savefig(image_path)

if __name__ == "__main__":
    
    # python predict.py --model_name pipeline_stock_predictor --model_version 2 --previous_time_window 10 --next_time_window 5 --vis

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="pipeline_stock_predictor")
    parser.add_argument("--model_version", type=int, required=False, default=1)
    parser.add_argument("--previous_time_window", type=int, required=False, default=10)
    parser.add_argument("--next_time_window", type=int, required=False, default=5)
    parser.add_argument("--vis", action='store_true')
    args = parser.parse_args()

    predict(args.model_name, args.model_version, args.previous_time_window, args.next_time_window, args.vis)
    