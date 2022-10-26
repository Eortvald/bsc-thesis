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
        assert self.p % 2 == 0, 'P must be an even positive integer'
        self.half_p = torch.tensor(int(p/2))
        self.L_diag = nn.Parameter(torch.rand(self.p))
        self.L_under_diag = nn.Parameter(torch.tril(torch.rand(self.p, self.p),-1))
        self.SoftPlus = nn.Softplus()

    def log_sphere_surface(self):
        log_surf_a = torch.lgamma(self.half_p) - torch.log(torch.tensor(2 * np.pi ** self.half_p))
        return log_surf_a

    def compose_L(self):
        """ Cholesky Component -> A """

        L_diag_pos_definite = self.SoftPlus(self.L_diag)  #this is only semidefinite...Need definite

        L = torch.tril(self.L_under_diag,-1) + torch.eye(self.p) * L_diag_pos_definite
        L_inv = torch.linalg.inv(L)

        print(f"Should be Identity:\n {L @ L_inv}")

        log_det_A = 2 * torch.sum(torch.log(L_diag_pos_definite))
        A_inv = L_inv @ L_inv.T

        return log_det_A, A_inv

    # Propability Density function
    def log_pdf(self,X):
        log_det_A, A_inv = self.compose_L()

        log_acg_pdf = self.log_sphere_surface() - 0.5 * log_det_A - self.half_p * torch.log(X @ A_inv.T @ X) # Matrix mult not finalized

        return log_acg_pdf

    def forward(self,X):
        return self.log_pdf(X)


if __name__ == "__main__":

    ACG = AngularCentralGaussian(p=4)
    X = torch.rand(10,4)
    ACG(X)



