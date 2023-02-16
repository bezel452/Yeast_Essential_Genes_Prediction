import os
import json
import pandas as pd

organisms = {"Y_lipolytica", "S_pombe", "S_cerevisiae", "P_pastoris", "C_albicans", "gene1", "gene2"}

maxn = -1

for i in organisms:
    with open("./Data/%s_include_ortholog.json" % i, "r") as f :
        data = json.load(f)
    for sub in data:
        if len(sub["protein sequence"]) > maxn:
            maxn = len(sub["protein sequence"])

print(maxn)

with open('M_data.json', 'r') as f:
    data = json.load(f)

print(len(data))