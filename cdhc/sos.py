import random
import numpy as np
import pandas as pd

from method import Method

random.seed(1)
np.random.seed(1)
x = pd.read_csv('data/sos.csv')
# Dropping the lexA column
x = x.iloc[:, :8]

m = Method(x)
m.iterative_fit()

with open('xrs/sos-edges.dat', 'w') as f:
    f.write(m.G.edges)
