from pymatbridge import Matlab

mlab = Matlab()
mlab.start()

"""results = mlab.run_code('a=34.56')
print(results)


var = mlab.get_variable('a')
print(f'{var} is of type {type(var)}')
"""
Z = [6,4,2]
dim= 90

#{'M':20, 'alpha':2, 'p':dim/2, 'q':1.5, 'x':Z }

for m in range(1,30):

    res = mlab.run_func('matlab/mhg15/mhg.m', m, 2, 0.5, dim/2, Z)
    print(res['result'])
