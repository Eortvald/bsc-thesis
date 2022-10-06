import scipy.special as sc
import numpy as np

p = 3


Z = np.random.rand(p)
print(Z)

c = sc.hyp1f1(1/2, p/2, Z)
print(c)