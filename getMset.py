import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


all_data = list()
essential_data = list()
non_essential_data = list()

with open('M_data.json', 'r') as f:
    data = json.load(f)

for subdata in data :
    essential_dict = dict()
    non_essential_dict = dict()

    if list(subdata.values())[0] == "E" :
        essential_dict[list(subdata.keys())[0]] = "E"
        essential_dict["gene sequence"] = subdata["gene sequence"]
        essential_dict["matrix"] = subdata["matrix"]
        essential_data.append(essential_dict)

    if list(subdata.values())[0] == "NE" :
        non_essential_dict[list(subdata.keys())[0]] = "NE"
        non_essential_dict["gene sequence"] = subdata["gene sequence"]
        non_essential_dict["matrix"] = subdata["matrix"]
        non_essential_data.append(non_essential_dict)

all_data = essential_data * 4 + non_essential_data

X_train, X_test = train_test_split(all_data, test_size=0.2)

with open('trainsetM.txt', 'w') as f :
    essential = {'E':1, 'NE':0}
    for train in X_train :
        keys = list(train.keys())
        values = list(train.values())
        f.write('>%s|%s|training|%s' % (keys[0],essential[values[0]],values[2]))
        f.write('\n')
        f.write(train["gene sequence"])
        f.write('\n')

    for test in X_test :
        keys = list(test.keys())
        values = list(test.values())
        f.write('>%s|%s|testing|%s' % (keys[0],essential[values[0]],values[2]))
        f.write('\n')
        f.write(test["gene sequence"])
        f.write('\n')