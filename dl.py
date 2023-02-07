import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

def read_code(file):
    encodings = []
    labels = []
    with open(file) as f:
        records = f.readlines()

    for line in records:
        array = line.strip().split('\t') if line.strip() != '' else None
        encodings.append(list(map(float, array[1:])))
        labels.append(float(array[0]))
    return encodings, labels

def calculate_acc(y_true, y_pred):
    predict = y_pred.ge(.5).view(-1)
    return (y_true == predict).sum().float() / len(y_true)

def round_tensor(t, decimal_places = 3):
    return round(t.item(), decimal_places)


x_train, y_train = read_code('training_code.txt')
x_test, y_test = read_code('testing_code.txt')

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = torch.from_numpy(x_train).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())

x_test = torch.from_numpy(x_test).float()
y_test = torch.squeeze(torch.from_numpy(y_test).float())

loss_fn = nn.BCELoss()

net = Net(x_train.shape[1])

optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

net = net.to(device)
loss_fn = loss_fn.to(device)

x_epo = []
y_tracc = []
y_teacc = []
y_trloss = []
y_teloss = []

for epoch in range(50000):
    y_pred = net(x_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = loss_fn(y_pred, y_train)

    if epoch % 100 == 0:
        train_acc = calculate_acc(y_train, y_pred)

        y_test_pred = net(x_test)
        y_test_pred = torch.squeeze(y_test_pred)

        test_loss = loss_fn(y_test_pred, y_test)
        test_acc = calculate_acc(y_test, y_test_pred)
        print(f'''epoch {epoch}
            Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
            Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
            Learning-rate: {optimizer.param_groups[0]['lr']}
            ''')
        x_epo.append(epoch)
        y_tracc.append(round_tensor(train_acc))
        y_teacc.append(round_tensor(test_acc))
        y_trloss.append(round_tensor(train_loss))
        y_teloss.append(round_tensor(test_loss))
    
    if epoch % 10000 == 0:    
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * ((50000 - epoch) / 50000)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

plt.plot(x_epo, y_tracc, c='tab:red', label="train")
plt.plot(x_epo, y_teacc, c='tab:blue', label="test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Processing of Training")
plt.legend()
plt.savefig("acc1.png")
plt.show()

plt.plot(x_epo, y_trloss, c='tab:red', label="train")
plt.plot(x_epo, y_teloss, c='tab:blue', label="test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Processing of Training")
plt.legend()
plt.savefig("loss1.png")
plt.show()

torch.save(net, "1.pth")

