import torch
import os
import numpy as np
from torch import nn, optim, backends
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from ..data.synthetic_generator import synthetic3D

### Synthetic data trials
# Train Mixture model

def train_mixture(MixtureModel, data, num_epoch):
    model = MixtureModel.train()
    train_loss = 0

    dataset_size = len(train_loader.dataset)

    for batch_num, (img, gt_hm) in enumerate(train_loader):
        # print(f'Min{torch.min(gt_hm)} | Max{torch.max(gt_hm)}')
        # print(f'Mean{torch.mean(img, dim=(0,2,3))} | STD{torch.std(img,dim=(0,2,3))}')
        # print(gt_hm[0])
        # Regeneration and loss
        img = img.to(device)
        gt_hm = gt_hm.to(device)
        pred_hm = model(img)
        loss = model.calc_loss(pred_hm, gt_hm)

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_num % 10 == 0:
            progress = (batch_num + 1) * len(img)
            print(
                f'Batch[{batch_num + 1}/{int(dataset_size / BSIZE)}]-loss:{loss.item():.6f}\t | [{progress}/{dataset_size}]')


    print(f'\t \t Train Error: Avg loss: {train_loss:.7f}')
    print(100 * '^')
    return train_loss


#Train Hidden Markov model


if __name__ == '__main__':

