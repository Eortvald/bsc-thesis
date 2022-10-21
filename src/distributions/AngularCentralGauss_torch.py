#%%
from scipy.special import factorial
import numpy as np
import torch
import torch.nn as nn



# Class for Angular-Central-Gaussian spherical ditribution
#Based on: "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
class AngularCentralGaussian(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p
        assert self.p % 2 == 0, 'P must be an even positive integer'
        self.half_p = int(p/2)
        self.L_diag = nn.Parameter(torch.rand(self.p))
        self.L_under_diag = nn.Parameter(torch.tril(torch.rand(self.p, self.p),-1))
        self.SoftPlus = nn.Softplus()

    def log_sphere_surface(self):
        log_surf_a = -torch.log(torch.tensor(2 * np.pi**(self.half_p))) - torch.lgamma(torch.tensor([self.half_p]))

        return log_surf_a

    def compose_A(self):
        L_diag_cap = self.SoftPlus(self.L_diag)  #Enshure Positive semi-definite

        L = torch.tril(self.L_under_diag,-1) + torch.eye(self.p) * L_diag_cap

        A = L @ L.T

        return A

    # Propability Density function
    def log_pdf(self,X):

        A = self.compose_A()

        ##### To LOG
        ### surface Area division?
        # torch.logdet(input)::: backward through logdet() will be unstable in when input doesn’t have distinct singular values.
        log_acg_pdf = torch.linalg.det(A)**-0.5 * (X.T @ torch.linalg.inv(A) @ X)

        return log_acg_pdf



