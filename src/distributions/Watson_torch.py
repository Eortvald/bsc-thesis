import torch
import torch.nn as nn
import numpy as np

class LogWatsonMultivariate(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """
    def __init__(self,p, para_init=None):
        super().__init__()

        self.p = p
        self.mu = nn.Parameter(torch.rand(self.p))
        self.kappa = nn.Parameter(torch.rand(self.p,self.p))

        if para_init is not None:
            print('Custom Initilization of')
            #for special initialization of patamters

    def log_kummer(self, a, c, kappa):
        # inspiration form Morten Bessel function
        # Gamma based? see Mardia A.18
        logKum = torch.log(torch.tensor(1))
        return logKum

    def log_norm_constant(self):
        logC =  torch.lgamma(torch.tensor([self.p/2])) \
               - torch.log(torch.tensor(2 * np.pi**(self.p/2))) \
               - self.log_kummer(0.5, self.p/2, self.kappa)

        return logC

    def log_pdf(self,X):
        # Why Transpose in the end? see LL_Torch_Watson
        logpdf = self.log_norm_constant() + self.kappa * torch.matmul(self.mu,X)**2
        return logpdf

    def forward(self,X):
        return self.log_pdf(X)




class TorchMixtureModel(nn.Module):
    def __init__(self, distribution:object, K:int):
        super().__init__()
        self.K = K
        self.cluster_dist = distribution
        self.pi = nn.Parameter(torch.ones(self.K)*1/self.K)
        self.
    def log_likelihood_mixture(self):


    def forward(self):


class Watson(nn.modules):


