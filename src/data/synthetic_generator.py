import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple


# torch.manual_seed(500)

def synthetic3D(pi,
                Sigmas,
                num_points: int = 1000,
                point_dim: int = 3,
                transition_matrix=None, as_array: bool = False):
    num_clusters = Sigmas.shape[0]
    print(f'Simulate {num_points} point from {num_clusters} of clusters')

    Lower_chol = torch.zeros(3, 3, 3)
    for idx, sig in enumerate(Sigmas):
        sig = 100 * sig
        Lower_chol[idx, :, :] = torch.linalg.cholesky(sig)

    # mixture assign and sample
    X = torch.zeros(num_points, point_dim)
    cluster_allocation = torch.zeros(num_points)
    for n in range(num_points):
        cluster_list = list(range(num_clusters))
        n_clust_id = int(np.random.choice(cluster_list, 1, p=pi))  # sample one cluster with pi probaility
        nx = Lower_chol[n_clust_id] @ torch.randn(point_dim)
        X[n] = nn.functional.normalize(nx, dim=0)
        cluster_allocation[n] = n_clust_id

    # r_thres = torch.rand(1)
    # print(r_thres)
    # print(torch.where(torch.cumsum(pi, dim=0) > r_thres, as_tuple=False))

    if as_array:
        return np.array(X), np.array(cluster_allocation)

    if not transition_matrix:
        pass
        print('Generating Synthetic HMM sequence')
    return X, cluster_allocation


if __name__ == '__main__':

    sig1 = torch.diag(torch.tensor([1, 1e-3, 1e-3]))
    sig2 = torch.eye(3) + 0.9 * (torch.ones(3) - torch.eye(3))
    sig3 = torch.diag(torch.tensor([1e-3, 1, 1])) \
           + 0.9 * torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    #print(torch.linalg.eigh(sig1))

    SIGMAs = torch.stack([sig1, sig2, sig3], dim=0)
    #print(SIGMAs)
    PI = [0.6, 0.2, 0.2]

    X, cluster_id = synthetic3D(pi=PI, Sigmas=SIGMAs, num_points=3000, as_array=True)

    print(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    id_2_color = {0: 'cyan', 1: 'green', 2: 'magenta'}
    id_2_colorcode = {0:(1.,1.,1.), 1:(0.5,0.5,0.5), 2:(0.,0.,0.)}

    label_color = [id_2_color[id] for id in cluster_id]


    ax.scatter(1, 1e-3, 1e-3, s=80, c='black')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=4, alpha=0.5, c=label_color)
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_zlabel('$z$', fontsize=15)
    plt.show()
