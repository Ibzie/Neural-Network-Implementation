import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()  # instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Pick a manual seed for randomization
torch.manual_seed(41)

# Create an instance of model
model = Model()

# Load the Iris dataset
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)

# Change last column from strings to integers
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

# Train Test Split/ Set X, y
X = my_df.drop('variety', axis=1).values
y = my_df['variety'].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()

# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train)

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train)

    # Keep Track of our losses
    losses.append(loss)

    if i % 10 == 0:
        print(f'Epoch {i} and Loss: {loss}')

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

# # Graph it out!
# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')

# Evaluate Model on Test Data Set (validate model on test set)
# Basically turn off back propagation
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

# Evaluate loss
print(f'CE Loss: {loss}')

# Classification
correct = 0
with torch.no_grad():
    for i in range(len(X_test)):
        y_val = model.forward(X_test[i])
        if y_val.argmax().item() == y_test[i]:
            correct += 1
        print(f'{i+1}) {str(y_val.detach().numpy())} \t {y_test[i]} \t {y_val.argmax().item()}')

print(f'We got {correct} correct!')

# Predictions for new data
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
with torch.no_grad():
    print("Predictions")
    print(model(new_iris).detach().numpy())

newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])
with torch.no_grad():
    print(model(newer_iris).detach().numpy())

# Save our NN Model
torch.save(model.state_dict(), 'my_really_awesome_iris_model.pt')

# Load the Saved Model
new_model = Model()
new_model.load_state_dict(torch.load('my_really_awesome_iris_model.pt'))
new_model.eval()
