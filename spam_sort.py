import csv
import torch
import torch.nn as nn
import torch.optim as optim
import os

print(os.getcwd())
print(os.listdir())

x = []
y = []

# Load CSV
with open('untitled folder/Messages.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    for row in reader:
        x.append(row[0])       # message
        y.append(int(row[1]))  # label (0 or 1)

# Keep messages as strings
X = x

# Convert labels into tensor
y = torch.tensor(y, dtype=torch.float32)

# 80/20 split
split_index = int(0.8 * len(X))

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Build vocabulary dictionary
vocab = {"UNKNOWN": 0}
next_index = 1

for message in X_train:
    for word in message.split():
        if word not in vocab:
            vocab[word] = next_index
            next_index += 1

# Convert training messages into numbers
X_train_encoded = []

for message in X_train:
    split_message = message.split()
    word_list = []

    for word in split_message:
        vocab_number = vocab[word]
        word_list.append(vocab_number)

    X_train_encoded.append(word_list)

# Padding step (next thing we were doing)
max_len = 0

for message in X_train_encoded:
    if len(message) > max_len:
        max_len = len(message)

for message in X_train_encoded:
    while len(message) < max_len:
        message.append(0)

X_train_tensor = torch.tensor(X_train_encoded, dtype = torch.float32)

class Detector(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, 7)
        self.fc2 = nn.Linear(7, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = Detector(max_len)

epochs = 1000

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.01
)

loss_function = nn.BCELoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = loss_function(predictions.squeeze(), y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    for message in X_test:
        split_message = message.split()
        word_list = []

        for word in split_message:
            if word in vocab:
                vocab_number = vocab[word]
            else:
                vocab_number = vocab["UNKNOWN"]
            word_list.append(vocab_number)

        while len(word_list) < max_len:
            word_list.append(0)

        if len(word_list) > max_len:
            word_list = word_list[:max_len]

        message_tensor = torch.tensor([word_list], dtype=torch.float32)
        prediction = model(message_tensor)
        pred_message = []
        if(prediction.item() > 0.5):
            pred_message.append("Spam")
        else:
            pred_message.append("Not Spam")
        print(f"Message: {message} | Prediction: {pred_message[0]}")
