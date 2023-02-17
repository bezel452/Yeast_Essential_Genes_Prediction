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
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, (10, 10))
        self.conv2 = torch.nn.Conv2d(10, 20, (10, 10))
        self.maxpooling = torch.nn.MaxPool2d(2)
        self.fc = nn.Linear(278560, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(4, 1)
    
    def forward(self, x, m):

        m = F.relu(self.maxpooling(self.conv1(m)))
        m = F.relu(self.maxpooling(self.conv2(m)))
        m = m.view(1, -1)
        x = torch.cat([x, m], 1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = x.squeeze(-1)
        return x


def read_code(file):
    encodings = []
    labels = []
    matrix = []
    with open(file) as f:
        records = f.readlines()

    for line in records:
        array = line.strip().split('\t') if line.strip() != '' else None
        encodings.append(list(map(float, array[1:-1])))
        labels.append(float(array[0]))
        matrix.append(array[-1])
    return encodings, labels, matrix

x_1, y_1, M_train = read_code('trainingM_code.txt')
x_2, y_2, M_test = read_code('testingM_code.txt')

x_train = []
y_train = []
x_test = []
y_test = []

xM_train = []
for mat in M_train:
    with open(mat, 'r') as f:
        l = f.readline()
        tmp = []
        data = l.split('\t')
        data = list(map(float, data))
        
        while l:
            tmp.append(data)
            data = l.split('\t')
            data = list(map(float, data))
            l = f.readline()
        xM_train.append(tmp)

xM_test = []
for mat in M_test:
    with open(mat, 'r') as f:
        l = f.readline()
        tmp = []
        data = l.split('\t')
        data = list(map(float, data))
        while l:
            tmp.append(data)
            data = l.split('\t')
            data = list(map(float, data))
            l = f.readline()
        xM_test.append(tmp)
Mat_train = []
Mat_test = []
cnt = -1
for data in xM_train:
    tmp = []
    cnt += 1
    if len(data) > 500:
        continue
    for i in range(0, 500):
        if len(data) > i:
            t = data[i]
            t = t + [0.0] * (500 - len(data[i]))
            tmp.append(t)
        else:
            tmp.append([0.0] * 500)
    Mat_train.append(tmp)
    x_train.append(x_1[cnt])
    y_train.append(y_1[cnt])

cnt = -1
for data in xM_test:
    tmp = []
    cnt += 1
    if len(data) > 500:
        continue
    for i in range(0, 500):
        if len(data) > i:
            t = data[i]
            t = t + [0.0] * (500 - len(data[i]))
            tmp.append(t)
        else:
            tmp.append([0.0] * 500)
    Mat_test.append(tmp)
    x_test.append(x_2[cnt])
    y_test.append(y_2[cnt])

Mat_train = torch.FloatTensor(Mat_train)
Mat_test = torch.FloatTensor(Mat_test)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = torch.from_numpy(x_train).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())

x_test = torch.from_numpy(x_test).float()
y_test = torch.squeeze(torch.from_numpy(y_test).float())

model = Net()
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)
Mat_train = Mat_train.to(device)
Mat_test = Mat_test.to(device)
loss_fn = loss_fn.to(device)

def train(epoch):
    running_loss = 0.0
    l = len(x_train)
    for i in range(0, l):
        optimizer.zero_grad()
        y_pred = model(torch.unsqueeze(x_train[i], 0), torch.unsqueeze(Mat_train[i], 0))
        loss = loss_fn(y_pred, torch.unsqueeze(y_train[i], 0))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 300 == 299:
            print("[%d,%5d] loss:%.3f"%(epoch+1, i + 1, running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = len(x_test)
    with torch.no_grad():
        l = len(x_test)
        for i in range(0, l):
            y_pred = model(torch.unsqueeze(x_test[i], 0), torch.unsqueeze(Mat_test[i], 0))
            y = torch.unsqueeze(y_test[i], 0)
        #    print(y)
            predict = y_pred.ge(.5).view(-1)
            correct += (predict == y).sum().float()
    print('Accuracy on Test set:%d%% [%d/%d]'%(100*correct/total, correct, total))
    accuracy_list.append(100*correct/total)


if __name__ == '__main__':
    accuracy_list = []
    for epoch in range(20):
        train(epoch)
        test()
    torch.save(model.state_dict(), '2.pt')
    