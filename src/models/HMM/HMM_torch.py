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
        self.transition_matrix = nn.Parameter(torch.ones(self.N, self.N) / self.N)
        self.emission_density = emission_dist
        self.obs_dim = observation_dim
        self.state_priors = nn.Parameter(torch.ones(self.N) * 1 / self.N)
        self.emission_models = nn.ModuleList([self.emission_density(self.obs_dim) for _ in range(self.N)])


    def emission_models_forward(self, X):
        return torch.stack([state_emission(X) for state_emission in self.emission_models])

    def transition_prop_forward(self, transition):
        pass

    def forward(self, X):
        """
        :param X: (num_subject/batch_size, observation_sequence, sample_x(dim=obs_dim))
        :return: log_prob
        """

        # init  1)
        log_A = nn.functional.log_softmax(self.transition_matrix)
        log_pi = nn.functional.log_softmax(self.state_priors)

        num_subjects = X.shape[0]
        seq_max = X.shape[1]
        log_alpha = torch.zeros(num_subjects, seq_max, self.N)

        # time t=0
        # log_pi: (n states priors)
        # emission forward return: -> transpose -> (subject, [state1_prop(x)...stateN_prop(x)])
        print(self.emission_models_forward(X[:, 0, :]).T)
        log_alpha[:, 0, :] = log_pi + self.emission_models_forward(X[:, 0, :]).T
        print(log_alpha[:, 0, :])

        # Induction 2)
        # for time:  t = 1 -> seq_max
        for t in range(1, seq_max):
            log_alpha[:, t, :] = self.emission_models_forward(X[:, t]) \
                                 + torch.logsumexp(log_alpha[:, t - 1, :, None] + log_A, dim=1)

        #### Termination 3)
        print(log_alpha[0, :, :])
        print(log_alpha[0, :, :].shape)
        log_prop = torch.logsumexp(log_alpha, dim=2)

        return log_prop  # return log_prop per subject

    def viterbi(self, X):
        # inspiraiton from https://colab.research.google.com/drive/1S__f14spx4BVbtNGIDL27th2NpijZjOl
        optimal_seq_Z = torch.tensor([1, 2, 3, 4, 5])

        return optimal_seq_Z


if __name__ == '__main__':
    from src.distributions.Watson_torch import Watson

    HMM = HiddenMarkovModel(num_states=6, observation_dim=3, emission_dist=Watson)
    X = torch.rand(2, 8, 3)  # num_subject, seq_max, observation_dim

    print(HMM(X))
