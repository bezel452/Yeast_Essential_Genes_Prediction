import os
import json
import pandas as pd

organisms = {"Y_lipolytica", "S_pombe", "S_cerevisiae", "P_pastoris", "C_albicans", "gene1", "gene2"}

all_data = list()
essential_data = list()
non_essential_data = list()

Mdata = []

for organism in organisms :
    # print("This organism is: %s" % organism)
    with open("./Data/%s_include_ortholog.json" % organism, "r") as f :
        data = json.load(f)
    for root, dir, files in os.walk('./MATX'):
        for file in files:
            file = file.split('.')
            tmp = {}
            
            for sub in data:
                for key in sub:
                    if len(file) == 2:
                        if key == file[0]:
                            tmp[key] = sub[key]
                            tmp["gene sequence"] = sub["gene sequence"]
                            tmp["protein sequence"] = sub["protein sequence"]
                            tmp['matrix'] = root + '/' + file[0] + ".txt"
                            Mdata.append(tmp)
                            break
                    elif len(file) == 3:
                        x = file[0] + '.' + file[1]
                        if key == x:
                            tmp[key] = sub[key]
                            tmp["gene sequence"] = sub["gene sequence"]
                            tmp["protein sequence"] = sub["protein sequence"]
                            tmp['matrix'] = root + '/' + x + ".txt"
                            Mdata.append(tmp)
                            break
                    elif len(file) == 4:
                        x = file[0] + '.' + file[1] + '.' + file[2]
                        if key == x:
                            tmp[key] = sub[key]
                            tmp["gene sequence"] = sub["gene sequence"]
                            tmp["protein sequence"] = sub["protein sequence"]
                            tmp['matrix'] = root + '/' + x + ".txt"
                            Mdata.append(tmp)
                            break

with open('M_data.json', 'w') as f:
    data = json.dumps(Mdata, sort_keys=True, indent=4, separators=(',', ': '))
    f.write(data)