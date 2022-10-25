import torch
import torch.nn as nn
import numpy as np


class Watson(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """
    def __init__(self, p):
        super().__init__()

        self.p = p
        self.mu = nn.Parameter(torch.rand(self.p))
        self.kappa = nn.Parameter(torch.tensor([1.]))
        self.SoftPlus = nn.Softplus()  # Log?
        self.const_a = torch.tensor(0.5)

    def log_kummer(self, a, b, kappa):


        n = torch.arange(10000)  # precicion order

        inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
                + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1))

        logkum = torch.logsumexp(inner, dim=0)
        return logkum

    def log_norm_constant(self):
        logC = torch.lgamma(torch.tensor(self.p / 2)) - torch.log(torch.tensor(2 * np.pi ** (self.p / 2))) \
               - self.log_kummer(self.const_a, torch.tensor(self.p / 2), self.kappa)  # addiction kummer?

        return logC

    def log_pdf(self, X):
        # Constraints
        kappa_positive = self.SoftPlus(self.kappa)

        mu_unit = nn.functional.normalize(self.mu, dim=0)
        print(f'Norm of mu:{mu_unit.norm()}')
        #assert torch.abs(mu_unit.norm() - 1.) > 1.e-3, "mu is not properly normalized"


        # log PDF
        logpdf = self.log_norm_constant() + kappa_positive * (mu_unit @ X.T) ** 2
        return logpdf

    def forward(self, X):
        return self.log_pdf(X)


if __name__ == "__main__":
    W = Watson(p=3)
    for p in iter(W.parameters()):
        print(p)
