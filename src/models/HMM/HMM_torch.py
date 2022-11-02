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
    def __init__(self, num_states, observation_dim, emission_dist):
        super(HiddenMarkovModel, self).__init__()

        self.N = num_states
        self.transition_matrix = nn.Parameter(torch.ones(self.N,self.N)/self.N)
        self.emission_density = emission_dist
        self.obs_dim = observation_dim
        self.state_priors = nn.Parameter(torch.ones(self.N) * 1 / self.N)
        self.emission_models = nn.ModuleList([self.emission_density(self.obs_dim) for _ in range(self.N)])

        self.LogSoftMax = nn.LogSoftmax()

    def emission_models_forward(self,X):
        return torch.stack([state_emission(X) for state_emission in self.emission_models])

    def transution_prop_forward(self,transition):


    def forward(self,X):
        #init
        log_A = self.LogSoftMax(self.transmission_matrix)
        log_pi = self.LogSoftMax(self.state_priors)

        num_subjects = X.shape[0]
        seq_max = X.shape[1]
        log_alpha = torch.zeros(num_subjects, seq_max, self.N)

        # time t=0
        log_alpha[:, 0, :] = log_pi + self.emission_models_forward(X[:, 0])

        # for time:  t = 1 -> seq_max
        for t in range(1,seq_max):
            log_alpha[:, t, :] = self.emission_models_forward(X[:, t])







    def viterbi(self,X):

        optimal_seq_Z = torch.tensor([1,2,3,4,5])


        return optimal_seq_Z




if __name__ == '__main__':

    from src.distributions.Watson_torch import Watson
    HMM = HiddenMarkovModel(num_states=6,observation_dim=3,emission_dist=Watson)
    for state in HMM.emission_models:
        print(state.kappa)
    X = torch.rand(10,3)
    print(HMM.emission_model_forward(X))