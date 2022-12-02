import torch
import os
import numpy as np
from torch import nn, optim, backends
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py

from tqdm.notebook import trange, tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train Mixture model
def train_mixture(MixtureModel, data, optimizer, num_epoch=100, keep_bar=True):
    model = MixtureModel.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)
    for epoch in tqdm(range(num_epoch), leave=keep_bar):

        leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood.item()


    return epoch_likelihood_collector


def train_mixture_subjects(MixtureModel, data, optimizer, num_epoch=100, print_progress=False):
    model = MixtureModel.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch)):
        epoch_LogLikelihood = 0
        for subject in data:
            all_leida_vectors = subject.to(device)

            subject_Likelihood = -model(all_leida_vectors)  # OBS! Negative

            epoch_LogLikelihood += subject_Likelihood
            optimizer.zero_grad()
            subject_Likelihood.backward()
            optimizer.step()

        epoch_likelihood_collector[epoch] = epoch_LogLikelihood

        if print_progress:
            print(100 * 'v')
            print(f'Epoch: {epoch + 1} \t | Negative LogLikelihood {epoch_LogLikelihood:.7f}')
            print(100 * '^')


    return epoch_likelihood_collector


# Train Hidden Markov model



def train_hmm(HMM, data, optimizer, num_epoch=100, print_progress=False):

    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch)):

        subject_leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(subject_leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood

        if print_progress:
            print(100 * 'v')
            print(f'Epoch: {epoch + 1} \t | Negative LogLikelihood {NegativeLogLikelihood:.7f}')
            print(100 * '^')

    return epoch_likelihood_collector


def train_hmm_subjects(HMM, data, optimizer, num_epoch=100, print_progress=False):
    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch)):

        epoch_LogLikelihood = 0
        for subject in data:
            subejct_leida_vectors_seq = subject.to(device)

            subject_Likelihood = -model(subejct_leida_vectors_seq)  # OBS! Negative

            epoch_LogLikelihood += subject_Likelihood
            optimizer.zero_grad()
            subject_Likelihood.backward()
            optimizer.step()

        epoch_likelihood_collector[epoch] = epoch_LogLikelihood


        subject_leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(subject_leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood

        if print_progress:
            print(100 * 'v')
            print(f'Epoch: {epoch + 1} \t | Negative LogLikelihood {NegativeLogLikelihood:.7f}')
            print(100 * '^')

    return epoch_likelihood_collector











if __name__ == '__main__':
    pass
