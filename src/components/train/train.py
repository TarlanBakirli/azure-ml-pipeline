import torch
import torch.nn as nn
import os
import numpy as np

from pathlib import Path

def get_files(f):

    f = Path(f)
    files = list(f.iterdir())
    return files

def train(input_data, output_model, epochs, learning_rate, previous_time_window, next_time_window):
    
    files = get_files(input_data)
    for file in files:
        if file.name == "X_train.npy":
            X_train = np.load(file)
        elif file.name == "y_train.npy":
            y_train = np.load(file)
        else:
            print("Unrecognized file")

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    model = nn.Sequential(
        nn.Linear(previous_time_window, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, next_time_window),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    for n in range(epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n % 10 == 0:
            print(f"Iteration {n}: ", loss.item())
    print("Training Done")

    print("Saving model...")
    model_path = os.path.join(output_model, "pipeline_stock_predictor_model.pt")
    torch.save(model, model_path)
    print(f"Model saved into {model_path}")