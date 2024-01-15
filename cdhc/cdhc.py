import random
import argparse

import numpy as np
import pandas as pd
import pymc3 as pm

from method import Method

parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='input_file', type=str)
parser.add_argument('--out', dest='output_file', type=str)
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

random.seed(1)
np.random.seed(1)
x = pd.read_csv(input_file)
print(x.shape)

# This is the only part which actually pertains to our method
m = Method(x)
m.iterative_fit()
edges = m.G.edges

with open(output_file, 'w') as f:
    f.write(m.G.edges)
