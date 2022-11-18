import torch
import os
import numpy as np
from torch import nn, optim, backends
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from ..data.synthetic_generator import synthetic3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Synthetic data trials
# Train Mixture model

def train_mixture(MixtureModel, data, optimizer, num_epoch=100):

    model = MixtureModel.to(device).train()

    dataset_size = data.shape[0]
    epoch_likelihood = np.zeros(num_epoch)

    for epoch in range(num_epoch):


        epoch_likelihood = 0

        for batch_num, leida_vectors in enumerate(data):

            leida_vectors = leida_vectors.to(device)
            # sqeuze batch dim of leida vectors


            NegativeLogLikelihood = -model(leida_vectors) #OBS! Negative

            optimizer.zero_grad()
            NegativeLogLikelihood.backward()
            epoch_likelihood += NegativeLogLikelihood.item()
            optimizer.step()

        epoch_likelihood[epoch] = epoch_likelihood

        print(100 * 'v')
        print(f'Epoch: {epoch+1} \t | Negative LogLikelihood {epoch_likelihood:.7f}')
        print(100 * '^')


#Train Hidden Markov model


if __name__ == '__main__':

