import torch
import torch.nn as nn
import numpy as np

class WatsonDistribution(nn.module):
    def __init__(self,p, para_init=None):
        super().__init__()
        self.p = p
        self.mu = nn.Parameter(torch.rand(self.p))
        self.kappa = nn.Parameter(torch.tensor(1.0))

        if para_init is not None:
            print('empty')
            #for special initialization of patamters


class TorchMixtureModel(nn.Module):
    def __init__(self, distribution:object, K:int):
        super().__init__()
        self.K = K
        self.cluster_dist = distribution


    def log_kummer(self):

    def log_C(self):

    def log_pdf(self):

    def log_likelihood(self):


    def forward(self):


class Watson(nn.modules):


