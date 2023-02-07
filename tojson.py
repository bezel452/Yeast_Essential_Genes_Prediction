import numpy as np
import pandas as pd
import json

D8 = []
D9 = []

f = open("geneSeq.txt", 'r')
line = f.readline()
fw1 = open("gene1.txt", 'w')
fw2 = open("gene2.txt", 'w')
tmp = ""

flag = -1

while line:
    if line[0] == '>':
        if flag == 0:
            tmp = tmp + '\n'
            fw1.write(tmp)
        #    print(tmp)
            tmp = ""
            flag = -1
        elif flag == 1:
            tmp = tmp + '\n'
            fw2.write(tmp)
        #    print(tmp)
            tmp = ""
            flag = -1
        if line[7] == '8':
            fw1.write(line)
            flag = 0
        else:
            fw2.write(line)
            flag = 1
    else:
        tmp = tmp + line[:-1]
    line = f.readline()
if flag == 0:
    fw1.write(tmp)
        #    print(tmp)
    tmp = ""
elif flag == 1:
    fw2.write(tmp)
        #    print(tmp)
    tmp = ""

f.close()
fw1.close()
fw2.close()

fp1 = open("AASeq_1.fasta", 'r')
fp2 = open("AAseq_2.fasta", 'r')
fg1 = open("gene1.txt", 'r')
fg2 = open("gene2.txt", 'r')

line1 = fp1.readline()
line2 = fg1.readline()

n = ""
tmp = {}
flag = -1
while line1 and line2:
    if line1[0] == '>':
        if flag != -1:
            D8.append(tmp)
        tmp = {}
        n = line1[1:-1]
        flag = 1
    else:
        tmp[n] = 'E'
        tmp["gene sequence"] = line2[:-1]
        tmp["protein sequence"] = line1[:-1]
    
    line1 = fp1.readline()
    line2 = fg1.readline() 

flag = -1
tmp = {}
line1 = fp2.readline()
line2 = fg2.readline()
while line1 and line2:
    if line1[0] == '>':
        if flag != -1:
            D9.append(tmp)
        tmp = {}
        n = line1[1:-1]
        flag = 1
    else:
        tmp[n] = 'E'
        tmp["gene sequence"] = line2[:-1]
        tmp["protein sequence"] = line1[:-1]
    
    line1 = fp2.readline()
    line2 = fg2.readline() 

fp1.close()
fp2.close()
fg1.close()
fg2.close()

with open("gene1.json", 'w') as f:
    data = json.dumps(D8, sort_keys=True, indent=4, separators=(',', ': '))
    f.write(data)

with open("gene2.json", 'w') as f:
    data = json.dumps(D9, sort_keys=True, indent=4, separators=(',', ': '))
    f.write(data)


