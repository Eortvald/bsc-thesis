import numpy as np
from numpy.linalg import eigh

# Function Returning the Leading Eigenvector of a nxn Matrix
def leadingEigVec(At) -> np.array:
    EigenValues, EigenVectors = eigh(At)
    LeV = EigenVectors[:, np.argmax(EigenValues)]

    return LeV

# Function Estimating the Coherence map of each time point and extracting the Leading Eigen Vector
def coherenceMap(Theta):
    N, T = Theta.shape

    LEiDA_Signal = np.zeros((N, T))

    # loop for each time point
    for t in range(T):
        At = np.zeros((N, N))
        CurrentSample = Theta[:, t]

        # loop for coherence between all dimension
        for j in range(N):
            for k in range(N):
                At[j, k] = np.cos(CurrentSample[j] - CurrentSample[k])

        # Leading Eigenvector of coherence matrix as new data-point
        LEiDA_Signal[:, t] = leadingEigVec(At)

    return LEiDA_Signal


