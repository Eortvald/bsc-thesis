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

        self.logsoftmax_transition = nn.LogSoftmax(dim=1)
        self.logsoftmax_prior = nn.LogSoftmax(dim=0)

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
        log_A = self.logsoftmax_transition(self.transition_matrix)
        log_pi = self.logsoftmax_prior(self.state_priors)
        num_subjects = X.shape[0]
        seq_max = X.shape[1]
        log_alpha = torch.zeros(num_subjects, seq_max, self.N)

        # time t=0
        # log_pi: (n states priors)
        # emission forward return: -> transpose -> (subject, [state1_prop(x)...stateN_prop(x)])
        log_alpha[:, 0, :] = log_pi + self.emission_models_forward(X[:, 0, :]).T

        # Induction 2)
        # for time:  t = 1 -> seq_max

        for t in range(1, seq_max):
            log_alpha[:, t, :] = self.emission_models_forward(X[:, t, :]).T \
                                 + torch.logsumexp(log_alpha[:, t - 1, :, None] + log_A, dim=1)

        # Termination 3)
        # print(log_alpha[:, :, :])
        print(log_alpha.shape)
        #print(log_alpha)

        # LogSum over states N for each time t.
        log_t_sums = torch.logsumexp(log_alpha, dim=2)
        #print(log_t_sums)
        # Retrive the alpha for the last time t in the seq.

        log_props = torch.gather(log_t_sums, dim=1, index=torch.tensor([[seq_max-1]] * num_subjects)).squeeze()
        #faster on GPU than just indexing...according to stackoverflow
        return log_props.sum()  # return log_prop per subject

    def viterbi(self, X):
        # inspiraiton from https://colab.research.google.com/drive/1S__f14spx4BVbtNGIDL27th2NpijZjOl
        optimal_seq_Z = torch.tensor([1, 2, 3, 4, 5])

        return optimal_seq_Z


if __name__ == '__main__':
    from src.distributions.Watson_torch import Watson

    dim = 90

    HMM = HiddenMarkovModel(num_states=15, observation_dim=dim, emission_dist=Watson)
    X = torch.rand(200, 350, dim)  # num_subject, seq_max, observation_dim

    out_prop = HMM(X)
    print(out_prop)
