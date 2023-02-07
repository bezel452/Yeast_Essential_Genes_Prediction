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


net = torch.load("1.pth")
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

x_test, y_test = read_code('testing_code.txt')
x_test = np.array(x_test)
y_test = np.array(y_test)

y_true = y_test
x_test = torch.from_numpy(x_test).float()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x_test = x_test.to(device)

net = net.to(device)
y_test_pred = net(x_test)
y_test_pred = torch.squeeze(y_test_pred)
y_pre = y_test_pred
y_pre = y_pre.cpu().detach().numpy()

fpr, tpr, thresholds = roc_curve(y_true, y_pre, pos_label=1)

a = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='red',
         lw=2, label='ROC curve (area = %0.2f)' % a)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("AUC-ROC figure")
plt.legend(loc="lower right")
plt.savefig("ROC1.png")
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
plt.savefig("PRC1.png")
plt.show()