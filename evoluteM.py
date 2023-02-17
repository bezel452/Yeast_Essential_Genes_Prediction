import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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


model = Net()
model.load_state_dict(torch.load('2.pt'))

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

x_2, y_2, M_test = read_code('testingM_code.txt')
x_test = []
y_test = []

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

Mat_test = []
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
Mat_test = torch.FloatTensor(Mat_test)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_true = y_test
x_test = torch.from_numpy(x_test).float()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_test = x_test.to(device)
Mat_test = Mat_test.to(device)
model = model.to(device)
y_pre = []
l = len(x_test)
for i in range(0, l):
    y_tmp = model(torch.unsqueeze(x_test[i], 0), torch.unsqueeze(Mat_test[i], 0))
    y_tmp = torch.squeeze(y_tmp)
    y_tmp = y_tmp.cpu().detach().numpy()
    y_pre.append(y_tmp)


fpr, tpr, thresholds = roc_curve(y_true, y_pre, pos_label=1)

a = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='red',
         lw=2, label='ROC curve (area = %0.2f)' % a)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("AUC-ROC figure")
plt.legend(loc="lower right")
plt.savefig("ROC2.png")
plt.show()

precision, recall, _=precision_recall_curve(y_true, y_pre)
PRC = average_precision_score(y_true, y_pre)

plt.figure()
plt.title('PR curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.step(recall, precision, color='b', label=' (RPC={:.4f})'.format(PRC))
plt.legend(loc='lower right')
plt.savefig("PRC2.png")
plt.show()