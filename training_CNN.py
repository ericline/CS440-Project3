import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# Dataset

class ShipDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.layout = torch.tensor(data["layout"], dtype=torch.float32)
        self.configurations = data["configurations"]
        self.height, self.width = self.layout.shape

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, idx):
        config = self.configurations[idx]
        bx, by = config["bx"], config["by"]
        rx, ry = config["rx"], config["ry"]
        t_val = torch.tensor(config["t_value"], dtype=torch.float32)

        # Create one-hot maps
        bot_map = torch.zeros((self.height, self.width), dtype=torch.float32)
        rat_map = torch.zeros((self.height, self.width), dtype=torch.float32)
        bot_map[bx][by] = 1.0
        rat_map[rx][ry] = 1.0

        # Stack channels: [layout, bot position, rat position]
        input_tensor = torch.stack([self.layout, bot_map, rat_map])  # [3, H, W]
        return input_tensor, t_val



# Model

class ShipCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 30 * 30, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze() 



# Training Loop

def train(model, dataloader, optimizer, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:2d}: Training Loss: {avg_loss:.6f}")



# Main
def main():
    dataset = ShipDataset("ship_1_data_with_layout.json")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ShipCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train(model, dataloader, optimizer, loss_fn, epochs=10)


if __name__ == "__main__":
    main()
