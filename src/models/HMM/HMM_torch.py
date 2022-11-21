import torch
import torch.nn as nn
import numpy as np


class HiddenMarkovModel(nn.Module):
    """
    Hidden Markov model w. Continues observation density
    """

    def __init__(self, num_states, observation_dim, emission_dist):
        super().__init__()

        self.N = num_states
        self.transition_matrix = nn.Parameter(torch.rand(self.N, self.N) / self.N)
        self.emission_density = emission_dist
        self.obs_dim = observation_dim
        self.state_priors = nn.Parameter(torch.rand(self.N) / self.N)
        self.emission_models = nn.ModuleList([self.emission_density(self.obs_dim) for _ in range(self.N)])
        self.softplus = nn.Softplus(beta=20, threshold=1)
        self.logsoftmax_transition = nn.LogSoftmax(dim=1)
        self.logsoftmax_prior = nn.LogSoftmax(dim=0)

    @torch.no_grad()
    def get_model_param(self):
        priors_softmax = nn.functional.softmax(self.state_priors.data.to(torch.float64), dim=0).to(torch.float32)
        mixture_param_dict = {'priors': priors_softmax}
        mixture_param_dict['Transition_matrix'] = nn.functional.softmax(self.transition_matrix.data.to(torch.float64), dim=1).to(torch.float32)
        for comp_id, comp_param in enumerate(self.emission_models):
            mixture_param_dict[f'emission_model_{comp_id}'] = comp_param.get_params()
        return mixture_param_dict

    def emission_models_forward(self, X):
        return torch.stack([state_emission(X) for state_emission in self.emission_models])


    def forward(self, X):
        """
                Forward algorithm for a HMM - Solving 'Problem 1'

        :param X: (num_subject/batch_size, observation_sequence, sample_x(dim=obs_dim))
        :return: log_prob
        """
        # see (Rabiner, 1989)
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
        # LogSum for states N for each time t.
        log_t_sums = torch.logsumexp(log_alpha, dim=2)

        # Retrive the alpha for the last time t in the seq, per subject
        log_props = torch.gather(log_t_sums, dim=1, index=torch.tensor([[seq_max-1]] * num_subjects)).squeeze()
        # faster on GPU than just indexing...according to stackoverflow

        return log_props.sum(dim=0)  # return sum of log_prop for all subjects

    #torch.no_grad()   # Disables backprop since this is a inferencng method
    def viterbi(self, X):
        """
            (Rabiner, 1989)
            :param X: (num_subject/batch_size, observation_sequence, sample_x(dim=obs_dim))
            :return: State sequence
            Structure inspired by https://github.com/lorenlugosch/pytorch_HMM
        """

        # init 1)
        log_A = self.logsoftmax_transition(self.softplus(self.transition_matrix))
        log_pi = self.logsoftmax_prior(self.softplus(self.state_priors)) # log_pi: (n states priors)
        num_subjects = X.shape[0]
        seq_max = X.shape[1]

        log_delta = torch.zeros(num_subjects, seq_max, self.N)
        psi = torch.zeros(num_subjects, seq_max, self.N, dtype=torch.int32) # intergers - state seqeunces

        # time t=0
        # emission forward return: -> transpose -> (subject, [state1_prop(x)...stateN_prop(x)])
        log_delta[:, 0, :] = log_pi + self.emission_models_forward(X[:, 0, :]).T

        # Recursion 2)
        # for time:  t = 1 -> seq_max
        for t in range(1, seq_max):
            max_value, max_state_indice = torch.max(log_delta[:, t - 1, :, None] + log_A, dim=2)

            log_delta[:, t, :] = self.emission_models_forward(X[:, t, :]).T + max_value
            psi[:, t, :] = max_state_indice

        # Termination 3) & Path backtracking 4)
        # max value and argmax at each time t, per subject.
        subjects_path_probs = torch.max(log_delta, dim=2)[0][:, -1]
        subjects_state_paths = []

        # Batches split up here for easier parallelization
        # https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
        for subject in range(num_subjects):
            _, subject_argmax_T = log_delta[subject, -1].max(dim=0)
            subject_path = [subject_argmax_T.item()]

            for t in range(seq_max-1, 0, -1):
                #print(subject_path)
                #print(t)
                previous_state = psi[subject, t, subject_path[0]].item()

                subject_path.insert(0, previous_state)

            subjects_state_paths.append(subject_path)

        # Log probs!!
        subjects_state_paths_ = np.array(subjects_state_paths)
        subjects_path_probs_ = np.array(subjects_path_probs.detach().cpu())
        return subjects_state_paths_, subjects_path_probs_

    def viterbi_2(self, X):
        """
            (Rabiner, 1989)
            :param X: (num_subject/batch_size, observation_sequence, sample_x(dim=obs_dim))
            :return: State sequence
            Structure inspired by https://github.com/lorenlugosch/pytorch_HMM
        """

        # init 1)
        log_A = self.logsoftmax_transition(self.transition_matrix)
        log_pi = self.logsoftmax_prior(self.state_priors)  # log_pi: (n states priors)
        num_subjects = X.shape[0]
        seq_max = X.shape[1]

        log_delta = torch.zeros(num_subjects, seq_max, self.N)
        psi = torch.zeros(num_subjects, seq_max, self.N, dtype=torch.int32)  # intergers - state seqeunces

        # time t=0
        # emission forward return: -> transpose -> (subject, [state1_prop(x)...stateN_prop(x)])
        log_delta[:, 0, :] = log_pi + self.emission_models_forward(X[:, 0, :]).T

        # Recursion 2)
        # for time:  t = 1 -> seq_max
        for t in range(1, seq_max):
            max_value, max_state_indices = torch.max(log_delta[:, t - 1, :, None] + log_A, dim=2)
            print(f'Delta:\n{log_delta[:, t - 1, :, None]}')
            print(f'A:\n{log_A}')
            print(f'Delta + A:\n {log_delta[:, t - 1, :, None] + log_A}')
            print('max val and max index')
            print(max_value, max_state_indices)
            print(10*'---')

            if t == 10:
                break
            log_delta[:, t, :] = self.emission_models_forward(X[:, t, :]).T + max_value
            psi[:, t, :] = max_state_indices
        print(psi)

        # Termination 3) & Path backtracking 4)
        # max value and argmax at each time t, per subject.
        subjects_path_probs = torch.max(log_delta, dim=2)[0][:, -1]
        subjects_state_paths = []

        # Batches/Subject split up here for easier parallelization
        for subject in range(num_subjects):
            _, subject_argmax_T = log_delta[subject, -1].max(dim=0)

            subject_path = [subject_argmax_T.item()] #Last state, with max probs
            psi_subject = psi[subject]

            for t in range(seq_max - 2, 0, -1):

                previous_state = psi_subject[t, subject_path[0]].item()

                subject_path.insert(0, previous_state)

            subjects_state_paths.append(subject_path)



        # Log probs!!
        subjects_state_paths_ = np.array(subjects_state_paths)
        subjects_path_probs_ = np.array(subjects_path_probs.detach().cpu())
        return subjects_state_paths_, subjects_path_probs_


if __name__ == '__main__':
    from src.distributions.Watson_torch import Watson

    torch.manual_seed(5)
    dim = 90

    HMM = HiddenMarkovModel(num_states=30, observation_dim=dim, emission_dist=Watson)
    X = torch.rand(2, 20, dim)  # num_subject, seq_max, observation_dim

    seq, probs = HMM.viterbi(X)

    print(seq)
    print(probs)