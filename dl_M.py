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
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(4, 1)
    
    def forward(self, x, m):
        batch_size = m.size(0)
        m = F.relu(self.maxpooling(self.conv1(m)))
        m = F.relu(self.maxpooling(self.conv2(m)))
        m = m.view(batch_size, -1)
        x = torch.cat([x, m], 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
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

x_train, y_train, M_train = read_code('trainingM_code.txt')
x_test, y_test, M_test = read_code('testingM_code.txt')

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = torch.from_numpy(x_train).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())

x_test = torch.from_numpy(x_test).float()
y_test = torch.squeeze(torch.from_numpy(y_test).float())

xM_train = []
for mat in M_train:
    with open(mat, 'r') as f:
        l = f.readline()
        tmp = []
        data = l.split('\t')
        data = list(map(float, data))
        while l:
            tmp.append(data)
            print(len(data))
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
'''
xM_train = torch.Tensor(xM_train).float()
xM_test = torch.Tensor(xM_test).float()

print(xM_test)
'''