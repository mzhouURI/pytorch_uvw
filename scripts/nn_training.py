import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import random


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNN, self).__init__()
        
        self.layer1 = nn.Linear(input_size, 16)  # First hidden layer
        self.layer2 = nn.Linear(16, 32)          # Second hidden layer
        self.layer3 = nn.Linear(32, 16)          # Third hidden layer
        self.output_layer = nn.Linear(16, output_size)  # Output layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # x = torch.relu(self.layer1(x))   # Apply ReLU activation after the first layer
        # x = torch.relu(self.layer2(x))   # Apply ReLU activation after the second layer
        # x = torch.relu(self.layer3(x))   # Apply ReLU activation after the third layer
        x = self.leaky_relu(self.layer1(x))   # Apply LeakyReLU after the first layer
        x = self.leaky_relu(self.layer2(x))   # Apply LeakyReLU after the second layer
        x = self.leaky_relu(self.layer3(x))   # Apply LeakyReLU after the third layer

        x = self.output_layer(x)         # No activation for output layer (regression task)
        return x

# Set random seed for reproducibility
seed = 42

# Python random module
random.seed(seed)

# NumPy random module
np.random.seed(seed)

# PyTorch random modules
torch.manual_seed(seed)  # For CPU

# Optional: CUDNN settings for determinism (only applies to GPU, so it's safe to include but won't affect CPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# Load the CSV file
df = pd.read_csv('training_data.csv')

# Split the data into inputs (features) and outputs (labels)
t_cmd = df.iloc[:, 4:8].values  # assuming first 4 columns are inputs
uvw = df.iloc[:, 1:4].values  # assuming the next 3 columns are outputs

print(t_cmd.shape)
print(uvw.shape)
X_tensor = torch.tensor(t_cmd, dtype=torch.float32)  # X is your input data
y_tensor = torch.tensor(uvw, dtype=torch.float32)  # y is your output data


dataset = TensorDataset(X_tensor, y_tensor)

val_size = int(0.2 * len(dataset))  # 20% for validation
train_size = len(dataset) - val_size  # The rest is for training

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for both training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model, define loss and optimizer
model = FeedForwardNN(input_size=4, output_size=3)  # 4 inputs and 3 outputs
criterion = nn.MSELoss()  # For regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


# Training the model
train_losses = []
val_losses = []

epochs = 50  # Set the number of epochs for training

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)  # Average loss for the epoch
    train_losses.append(epoch_train_loss)  # Store the training loss for plotting

    print(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss}")

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(val_loader)  # Average loss for the epoch
    val_losses.append(epoch_val_loss)  # Store the validation loss for plotting

    # Clear the output and plot again after each epoch
    plt.clf()  # Clear the current figure
    plt.plot(range(epoch + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(epoch + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)  # Pause for a short time to update the plot


    # Optionally print the losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")




##testing

df = pd.read_csv('test_data.csv')

# Split the data into inputs (features) and outputs (labels)
x_test = df.iloc[:, 4:8].values  # assuming first 4 columns are inputs
y_test = df.iloc[:, 1:4].values  # assuming the next 3 columns are outputs

X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create a DataLoader for the test data (if you want batching, otherwise just use the tensors directly)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predictions = []
actuals = []


with torch.no_grad():
    for inputs, targets in test_dataloader:
        # Get model predictions
        outputs = model(inputs)
        
        # Store the predictions and actual targets
        predictions.append(outputs.numpy())  # Convert the tensor to numpy array
        actuals.append(targets.numpy())  # Convert the tensor to numpy array

# Convert the lists to numpy arrays
predictions_np = np.concatenate(predictions, axis=0)
actuals_np = np.concatenate(actuals, axis=0)

# Plot the predictions vs actuals for each output variable
plt.figure(2)  # Figure 2

# For each output (assuming 3 outputs in this case)
for i in range(predictions_np.shape[1]):  # predictions_np.shape[1] is the number of outputs (3)
    plt.subplot(1, 3, i+1)  # Create a subplot for each output
    plt.plot(actuals_np[:, i], label='Actual', color='blue', linestyle='dashed')
    plt.plot(predictions_np[:, i], label='Predicted', color='red')
    plt.title(f'Output {i+1}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.show()