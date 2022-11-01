import torch
import torch.nn as nn
import numpy as np



class EmissionModel(nn.Module):
    """
    Emission model for Markov state observation probability

    !Continues density emission
    """
    def __int__(self, distribution):
        super().__init__()


class HiddenMarkovModel(nn.Module):
    """
    Hidden Markov model w. Continues observation density
    """
    def __init__(self, num_states, observation_dim, emmision_dist):
        super(HiddenMarkovModel, self).__init__()

        self.N = num_states
        self.transmission_matrix = nn.Parameter(torch.ones(self.N,self.N)/self.N)
        self.emission_density = emmision_dist
        self.obs_dim = observation_dim
        self.state_priors = nn.Parameter(torch.ones(self.K) * 1 / self.K)
        self.emission_prop = nn.ModuleList([self.emission_density(self.obs_dim) for _ in range(self.N)])

        self.LogSoftMax = nn.LogSoftmax()


    def forward(self,X):
        A = self.LogSoftMax(self.transmission_matrix)
        pi_n = self.LogSoftMax(self.state_priors)





    def viterbi(self,X):

        optimal_seq_Z = torch.tensor([1,2,3,4,5])


        return optimal_seq_Z

