import numpy as np


def provoke_error():
    if np.random.rand() <= 0.5:
        assert(4 == 7)

for i in range(20):
    try:
        provoke_error()
    except:
        print(f'Run {i} was resulted in an error')
