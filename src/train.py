import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
# import mlflow

from sklearn.preprocessing import MinMaxScaler

class StockPredictor:
    def __init__(self, previous_time_window, next_time_window, num_epochs, learning_rate, output_dir):
        self.previous_time_window = previous_time_window
        self.next_time_window = next_time_window
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.model = nn.Sequential(
            nn.Linear(self.previous_time_window, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.next_time_window),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def generate_synthetic_data(self, n_points=1000, initial_price=100, volatility=0.02):
        '''
            Assume generated sequence corresponds to prices of the stock for the last 1000 days.
            Therefore, last N elements in the sequence correspond to last N days up to now.
        '''
        # Generate random returns using normal distribution
        returns = np.random.normal(loc=0.0001, scale=volatility, size=n_points)
        
        # Calculate price path using cumulative returns
        price_path = initial_price * np.exp(np.cumsum(returns))

        return price_path

    def preprocess_data(self, sequence):

        # Normalize data
        sequence = sequence.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(sequence)
        normalized_sequence = scaler.transform(sequence)
        normalized_sequence = normalized_sequence.flatten()

        # Split normalized sequence into chunks
        X, y = [], []
        for i in range(len(normalized_sequence) - self.previous_time_window - self.next_time_window + 1):
            X.append(normalized_sequence[i:(i + self.previous_time_window)])
            y.append(normalized_sequence[(i + self.previous_time_window):(i + self.previous_time_window + self.next_time_window)])
        
        # Convert to Tensor
        X, y = torch.Tensor(X), torch.Tensor(y)

        return X, y
    
    def train(self, X, y):
        for n in range(self.num_epochs):
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if n % 10 == 0:
                print(f"Iteration {n}: ", loss.item())
        print("Training Done")

        print("Saving model")
        os.makedirs(self.output_dir, exist_ok=True)
        model_path = os.path.join(self.output_dir, "stock_predictor_model.pt")
        torch.save(self.model, model_path)
        print(f"Model saved into {model_path}")

        return model_path

    def predict(self, X, trained_model_path):

        model_path = os.path.join(trained_model_path)
        model = torch.load(model_path, weights_only=False)
        model.eval()

        print("Predicting...")

        X = X[-1]
        y_pred = model(X)
        y_pred = y_pred.detach().numpy()

        print("Predicted: ", y_pred)

        # Visulize / save plot as image (AzureML monitors ./outputs/images directory continously to show images in the Images section of the job)
        dates = np.arange(self.next_time_window)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_pred, color='blue', linewidth=1)
        plt.title('Stock Prices for Next 5 days', fontsize=14, pad=15)
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        # plt.show()
        images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, f"next_{self.next_time_window}_days_stock_prices.png")
        plt.savefig(image_path)
        return y_pred

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--previous_time_window", type=int, required=False, default=10)
    parser.add_argument("--next_time_window", type=int, required=False, default=5)
    parser.add_argument("--num_epochs", type=int, required=False, default=1)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    stock_predictor = StockPredictor(args.previous_time_window, args.next_time_window, args.num_epochs, args.learning_rate, args.output_dir)

    # Generate data
    sequence = stock_predictor.generate_synthetic_data()

    # Preprocess data
    X, y = stock_predictor.preprocess_data(sequence)

    # Train
    model_path = stock_predictor.train(X, y)
    
    # Predict
    stock_predictor.predict(X, model_path)