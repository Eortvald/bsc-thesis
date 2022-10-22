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

    def log_kummer(self, a, b, kappa):
        
        n = torch.arange(10000)  # precicion order

        inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
                + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1))

        logkum = torch.logsumexp(inner, dim=0)
        return logkum

    def log_norm_constant(self):
        logC = torch.lgamma(torch.tensor([self.p / 2])) - torch.log(torch.tensor(2 * np.pi ** (self.p / 2))) \
               - self.log_kummer(0.5, self.p / 2, self.kappa)  # addiction kummer?

        return logC

    def log_pdf(self, X):
        # Constraints
        kappa_positive = self.SoftPlus(self.kappa)
        mu_unit = nn.functional.normalize(self.mu)

        # log PDF
        logpdf = self.log_norm_constant() + kappa_positive * torch.matmul(mu_unit, X.T) ** 2
        return logpdf

    def forward(self, X):
        return self.log_pdf(X)


class TorchMixtureModel(nn.Module):
    def __init__(self, distribution_object, K: int, dist_dim=90):
        super().__init__()
        self.distribution, self.K, self.p = distribution_object, K, dist_dim

        self.pi = nn.Parameter(torch.ones(self.K) * 1 / self.K)
        self.LogSoftMax = nn.LogSoftmax()
        self.mix_components = nn.ModuleList([self.distribution(self.p) for _ in range(self.K)])

    def log_likelihood_mixture(self, X):
        inner = self.LogSoftMax(self.pi) + torch.tensor([K_mixture(X) for K_mixture in self.mix_components])
        logLikeLihood = torch.logsumexp(inner, dim=0)  # correct dim?

        return logLikeLihood

    def forward(self, X):
        return self.log_likelihood_mixture(X)


if __name__ == "__main__":
    W = Watson(p=3)
    for p in iter(W.parameters()):
        print(p)
