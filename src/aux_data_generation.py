"""
generate auxillary data for model training and evaluation, including:
    ancestors.pk: ancestors array of icd codes
"""

import data_utils as data_utils
import numpy as np

output_data_dir = "/"

parent_to_child_file = "../aux_data/MIMIC_parentTochild"

I = 7042
root = 5367
# Load up the tree
PtoC = [[] for i in range(I)]

f = open(parent_to_child_file)
for line in f:
    line = [int(x) for x in line.strip().split('|')]
    PtoC[line[0]].append(line[1])
f.close()

# Create ancestors array
ancestors = [[] for i in range(I)]
children = PtoC[root]
for child in children:
    ancestors[child].append(root)
while len(children) > 0:
    new_children = []
    for child in children:
        for gc in PtoC[child]:
            ancestors[gc].extend([child] + ancestors[child])
            new_children.append(gc)
    children = new_children
for i in range(len(ancestors)):
    ancestors[i] = np.array(ancestors[i] + [i])

data_utils.save_obj(ancestors, output_data_dir+"ancestors.pk")