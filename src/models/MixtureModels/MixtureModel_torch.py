import torch
import torch.nn as nn
import numpy as np
from src.distributions.Watson_torch import Watson
from src.distributions.AngularCentralGauss_torch import AngularCentralGaussian



class TorchMixtureModel(nn.Module):
    def __init__(self, distribution_object, K: int, dist_dim=90):
        super().__init__()

        self.distribution, self.K, self.p = distribution_object, K, dist_dim
        self.pi = nn.Parameter(torch.ones(self.K) * 1 / self.K)
        self.mix_components = nn.ModuleList([self.distribution(self.p) for _ in range(self.K)])

        self.LogSoftMax = nn.LogSoftmax(dim=0)

    def log_likelihood_mixture(self, X):

        inner_pi = self.LogSoftMax(self.pi)[:,None]
        inner_pdf = torch.stack([K_comp_pdf(X) for K_comp_pdf in self.mix_components])

        inner = inner_pi + inner_pdf

        loglikelihood_x_i = torch.logsumexp(inner, dim=0) #Log likelihood over a sample of p-dimensional vectors
        #print(loglikelihood_x_i)
        logLikeLihood = torch.sum(loglikelihood_x_i)
        #print(logLikeLihood)
        return logLikeLihood

    def forward(self, X):
        return self.log_likelihood_mixture(X)


if __name__ == "__main__":
    torch.set_printoptions(precision=4)
    MW = TorchMixtureModel(Watson,K=2,dist_dim=3)

    data = torch.rand(6,3)
    print(data)
    MW(data)
