import torch
import os
import numpy as np
from torch import nn, optim, backends
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train Mixture model
def train_mixture(MixtureModel, data, optimizer, num_epoch=100, print_progress=False):
    model = MixtureModel.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch)):

        leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(leida_vectors)  # OBS! Negative

        optimizer.zero_grad()
        NegativeLogLikelihood.backward()
        optimizer.step()

        epoch_likelihood_collector[epoch] = NegativeLogLikelihood

        if print_progress:
            print(100 * 'v')
            print(f'Epoch: {epoch + 1} \t | Negative LogLikelihood {NegativeLogLikelihood:.7f}')
            print(100 * '^')

    return epoch_likelihood_collector
# Train Hidden Markov model



def train_hmm(HMM, data, optimizer, num_epoch=100, print_progress=False):

    model = HMM.to(device).train()

    epoch_likelihood_collector = np.zeros(num_epoch)

    for epoch in tqdm(range(num_epoch)):

        leida_vectors = data.to(device)

        NegativeLogLikelihood = -model(leida_vectors)  # OBS! Negative

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
