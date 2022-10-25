import torch
import torch.nn as nn
import numpy as np




class TransitionModel(nn.Module):
    """
    Matrix for transition probability between Markov state
    """

    def __init__(self, ):
        super().__init__()


class EmissionModel(nn.Module):
    """
    Emission model for Markov state observation probability

    !Continues density emission
    """
    def __int__(self):
        super().__init__()


class HiddenMarkovModel(nn.Module):
    """
    Hidden Markov model w. Continues observation density
    """
    def __init__(self, num_states, observation_dim, emmision_dist):
        super(HiddenMarkovModel, self).__init__()
        self.num_state = num_states
        self.obs_dim = observation_dim
        self.emission_model = emmision_dist
        self.state_priors = nn.Parameter(torch.rand())
        self.SoftPlus = nn.Softplus()  # Log????
        self.LogSoftMax = nn.LogSoftmax()


    def forward(self,X):

        kappa_positive = self.SoftPlus(self.kappa)
        mu_unit = nn.functional.normalize(self.mu)



    def viterbi(self,X):

        optimal_seq_Z = torch.tensor([1,2,3,4,5])


        return optimal_seq_Z

