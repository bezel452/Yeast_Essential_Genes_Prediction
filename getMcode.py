import argparse
import sys, os, re
import numpy as np
from pubscripts import save_file
from itertools import combinations
from descnucleotide import *

def read_file(file):
    with open(file) as f:
        records = f.read()
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        Mat = header_array[3]
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train, Mat])
    return fasta_sequences

parameters = {
        'Sequence_Type': 'DNA',
        'Sequence_File': 'trainsetM.txt',
        'Kmer_Size': 3,
        'Method': "DNC;Kmer",
        'Output_Format': "tsv",
    }

fastas = read_file(parameters['Sequence_File'])
cmd_coding = {
        'Kmer': ['Kmer.Kmer(training_data, k=%s, **kw)' % parameters['Kmer_Size'], 'Kmer.Kmer(testing_data, k=%s, **kw)' % parameters['Kmer_Size']],
        'DNC': ['DNC.DNC(training_data, **kw)', 'DNC.DNC(testing_data, **kw)'],
    }

kw = {'nclusters': 3, 'sof': 'sample', 'order': ''}
kw['order'] = 'ACGT' if parameters['Sequence_Type'] == 'DNA' or parameters['Sequence_Type'] == 'RNA' else 'ACDEFGHIKLMNPQRSTVWY'


training_data = []
testing_data = []
for sequence in fastas:
    if sequence[3] == 'training':
        training_data.append(sequence)
    else:
        testing_data.append(sequence)

training_code_dict = {}
testing_code_dict = {}
method_array = parameters['Method'].split(';')
for method in method_array:
    training_code_dict[method] = eval(cmd_coding[method][0])
    testing_code_dict[method] = eval(cmd_coding[method][1])
Mat_train = []
Mat_train.append(['matrix'])
Mat_test = []
Mat_test.append(['matrix']) 

for sub in training_data:
    tmp = [sub[4]]
    Mat_train.append(tmp)

for sub in testing_data:
    tmp = [sub[4]]
    Mat_test.append(tmp)

training_code = np.array(training_code_dict[method_array[0]])
testing_code = np.array(testing_code_dict[method_array[0]])

for i in range(1, len(method_array)):
    if training_code_dict[method_array[i]] != 0:
        training_code = np.concatenate((training_code, np.array(training_code_dict[method_array[i]])[:, 2:]), axis=1)
    if testing_code_dict[method_array[i]] != 0:
        testing_code = np.concatenate((testing_code, np.array(testing_code_dict[method_array[i]])[:, 2:]), axis=1)

training_code = np.concatenate((training_code, np.array(Mat_train)), axis=1)
testing_code = np.concatenate((testing_code, np.array(Mat_test)), axis=1)

training_code = training_code.tolist()
save_file.save_file(training_code, format=parameters['Output_Format'], file='trainingM_code.txt')
testing_code = testing_code.tolist()
save_file.save_file(testing_code, format=parameters['Output_Format'], file='testingM_code.txt')