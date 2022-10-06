import numpy as np

from hypergeomat import hypergeomPQ
import math
Z = [6,4,2]

p= 3
s = hypergeomPQ(m=20, a=[0.5], b=[p/2], x=Z, alpha=2)

print(s)