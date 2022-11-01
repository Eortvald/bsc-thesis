import numpy as np
import torch
import torch.nn as nn


class AngularCentralGaussian(nn.Module):
    """
    Angular-Central-Gaussian spherical distribution:

    "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
    """
    def __init__(self, p):
        super().__init__()

        self.p = p
        #assert self.p % 2 == 0, 'P must be an even positive integer'
        self.half_p = torch.tensor(p/2)
        self.L_diag = nn.Parameter(torch.rand(self.p))
        self.L_under_diag = nn.Parameter(torch.tril(torch.rand(self.p, self.p),-1))
        self.SoftPlus = nn.Softplus(beta=20,threshold=1)

    def log_sphere_surface(self):
        log_surf_a = torch.lgamma(self.half_p) - torch.log(2 * np.pi ** self.half_p)

        #log_w_p <- log(2) + 0.5 * p * log(pi) - lgamma(0.5 * p) ???
        return log_surf_a

    def compose_L(self):
        """ Cholesky Component -> A """

        L_diag_pos_definite = torch.log(self.SoftPlus(self.L_diag))  #this is only semidefinite...Need definite

        L = torch.tril(self.L_under_diag,-1) + torch.diagflat(L_diag_pos_definite)
        L_inv = torch.linalg.inv(L)

        #print(f"Should be Identity:\n {L @ L_inv}")

        log_det_A = 2 * torch.sum(torch.log(L_diag_pos_definite))
        A_inv = L_inv @ L_inv.T

        return log_det_A, A_inv

    # Propability Density function
    def log_pdf(self,X):
        log_det_A, A_inv = self.compose_L()

        if self.p == 1:
            log_acg_pdf = self.log_sphere_surface() - 0.5 * log_det_A - self.half_p * torch.log(X * A_inv * X)
        else:
            log_acg_pdf = self.log_sphere_surface() - 0.5 * log_det_A - self.half_p * torch.log(torch.diag(X @ A_inv @ X.T))

        return log_acg_pdf


    def forward(self,X):
        return self.log_pdf(X)


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('Qt5Agg')


    ACG = AngularCentralGaussian(p=2)
    # phi = linspace(0, 2 * pi, 320);
    # x = [cos(phi);sin(phi)];
    #

    ACG.L_under_diag = nn.Parameter(torch.ones(2,2))
    ACG.L_diag = nn.Parameter(torch.tensor([21.,2.5]))
    phi = torch.arange(0, 2*np.pi, 0.001)
    phi_arr =  np.array(phi)
    x = torch.column_stack((torch.cos(phi),torch.sin(phi)))

    points = torch.exp(ACG(x))
    props = np.array(points.squeeze().detach())


    #props = props/np.max(props)

    ax = plt.axes(projection='3d')
    ax.plot(np.cos(phi_arr), np.sin(phi_arr), 0, 'gray')
    ax.scatter(np.cos(phi_arr), np.sin(phi_arr), props, c=props, cmap='viridis', linewidth=0.5)

    ax.view_init(30, 135)
    plt.show()
    #plt.show()
    # plt.scatter(phi,props, s=3)
    # plt.show()


