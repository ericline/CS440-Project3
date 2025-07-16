import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load CSV
df = pd.read_csv("ship_3_data.csv")

# Normalize input features and targets
features = df[["bx", "by", "rx", "ry"]].values / 30.0
raw_targets = df["t_value"].values.reshape(-1, 1)
t_min = raw_targets.min()
t_max = raw_targets.max()
targets = (raw_targets - t_min) / (t_max - t_min)

# Convert to tensors
x_data = torch.tensor(features, dtype=torch.float32)
y_data = torch.tensor(targets, dtype=torch.float32)

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=310)

# Neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 320)
        self.fc2 = nn.Linear(320, 128)
        self.fc3 = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

model = Net()

# Training setup
alpha = 0.01
batch_size = 640
epochs = 50
train_losses = []
test_losses = []

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
criterion = nn.MSELoss()

# DataLoader for batching and shuffling
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        test_predictions = model(x_test)
        test_mse = criterion(test_predictions, y_test).item()
        test_losses.append(test_mse)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_mse:.6f}")

# Final RMSE in original scale
test_rmse = np.sqrt(test_losses[-1])
test_rmse_actual = test_rmse * (t_max - t_min)
print(f"\nFinal Normalized RMSE: {test_rmse:.4f}")
print(f"Final Actual RMSE: {test_rmse_actual:.2f}")

# Plot loss curves
plt.plot(range(epochs), train_losses, label="Training Loss", marker='o')
plt.plot(range(epochs), test_losses, label="Testing Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Testing Loss")
plt.legend()
plt.grid(True)
plt.show()



