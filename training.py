import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("ship_1_data.csv")

# Normalize input features to (-1, 1)
features_raw = df[["bx", "by", "rx", "ry"]].values
f_min = features_raw.min(axis=0)
f_max = features_raw.max(axis=0)
features = 2 * (features_raw - f_min) / (f_max - f_min) - 1

# Normalize targets to (-1, 1)
raw_targets = df["t_value"].values.reshape(-1, 1)
t_min = raw_targets.min()
t_max = raw_targets.max()
targets = 2 * (raw_targets - t_min) / (t_max - t_min) - 1

# Histogram of raw t_value
plt.figure(figsize=(6, 4))
plt.hist(df["t_value"], bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of Original t_value")
plt.xlabel("t_value")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert to tensors
x_data = torch.tensor(features, dtype=torch.float32)
y_data = torch.tensor(targets, dtype=torch.float32)

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=310)

def get_batch(x, y, batch_size):
    n = x.shape[0]
    batch_indices = random.sample(range(n), k=batch_size)
    x_batch = x[batch_indices]
    y_batch = y[batch_indices]
    return x_batch, y_batch

# Neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=4, out_features=128, bias=True)
        self.layer_2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        return self.out(x)

model = Net()
print(model)

# Evaluate initial model before training
with torch.no_grad():
    initial_predictions = model(x_test).numpy()
    initial_true_values = y_test.numpy()

initial_predictions_actual = (initial_predictions + 1) / 2 * (t_max - t_min) + t_min
initial_true_values_actual = (initial_true_values + 1) / 2 * (t_max - t_min) + t_min

plt.figure(figsize=(6, 6))
plt.scatter(initial_true_values_actual, initial_predictions_actual, alpha=0.6, color='salmon', edgecolor='black')
plt.plot([initial_true_values_actual.min(), initial_true_values_actual.max()],
         [initial_true_values_actual.min(), initial_true_values_actual.max()],
         color='blue', linestyle='--', label='Ideal Prediction')
plt.xlabel("True t_value")
plt.ylabel("Predicted t_value")
plt.title("Initial Model: Predicted vs True t_value (Actual Scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Training setup
alpha = 0.0001
batch_size = 320
epochs = 10
train_losses = []
test_losses = []

optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0.0
    num_batches = len(x_train) // batch_size

    for _ in range(num_batches):
        x_batch, y_batch = get_batch(x_train, y_train, batch_size)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / num_batches
    train_losses.append(avg_train_loss)

    with torch.no_grad():
        test_predictions = model(x_test)
        test_mse = criterion(test_predictions, y_test).item()
        test_losses.append(test_mse)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_mse:.6f}")

# Final RMSE
test_rmse = np.sqrt(test_losses[-1])
test_rmse_actual = test_rmse * (t_max - t_min) / 2
# print(f"Final Normalized RMSE: {test_rmse:.4f}")
# print(f"Final Actual RMSE: {test_rmse_actual:.2f}")

# Plot loss curves
plt.plot(range(epochs), train_losses, label="Training Loss", marker='o')
plt.plot(range(epochs), test_losses, label="Testing Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.legend()
plt.grid(True)
plt.show()

# Final prediction scatter plot
with torch.no_grad():
    final_predictions = model(x_test).numpy()
    true_values = y_test.numpy()

final_predictions_actual = (final_predictions + 1) / 2 * (t_max - t_min) + t_min
true_values_actual = (true_values + 1) / 2 * (t_max - t_min) + t_min

plt.figure(figsize=(6, 6))
plt.scatter(true_values_actual, final_predictions_actual, alpha=0.6, color='mediumseagreen', edgecolor='black')
plt.plot([true_values_actual.min(), true_values_actual.max()],
         [true_values_actual.min(), true_values_actual.max()],
         color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("True t_value")
plt.ylabel("Predicted t_value")
plt.title("Final Model: Predicted vs True t_value (Actual Scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




