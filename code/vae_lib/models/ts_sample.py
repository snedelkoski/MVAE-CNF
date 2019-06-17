#train MVAE on a sample of the data just for a test
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
# from code.vae_lib.models.MVAEGAN import MCVAEGAN
from code.vae_lib.models.seq_mvae import SeqMVAE
from code.vae_lib.models.model import StandardNormalization

from ml_utils.iap_loader import IAPDataset
from code.vae_lib.models.train import load_checkpoint
from code.vae_lib.models.train_misc import override_divergence_fn


def fetch_ts_sample():
    """Return a random image from the MNIST dataset with label.

    @param label: integer
                  a integer from 0 to 9
    @return: torch.autograd.Variable
             MNIST image
    """
    ts_dataset = IAPDataset('/home/MihailBogojeski/git/multimodaldiffeq/data/basf-iap/synthetic/',
                            train=False, target_id=0)
    condition, target, length = ts_dataset[np.random.randint(len(ts_dataset))]
    return condition, target, length


def transform_to_sequence(sel_inputs, total_length):
    seq_inputs = []
    for i in range(total_length):
        curr_inp = [None] * len(sel_inputs)
        for j in range(len(sel_inputs)):
            if sel_inputs[j] is not None:
                # print('sel input shape', sel_inputs[j].shape)
                # print('total_length', total_length)
                if sel_inputs[j].shape[1] == total_length:
                    curr_inp[j] = sel_inputs[j][:, i, :]
                else:
                    curr_inp[j] = sel_inputs[j][:, 0, :]
        seq_inputs.append(curr_inp)
    return(seq_inputs)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../trained_models/checkpoint.pth.tar', help='path to trained model file')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of images and texts to sample [default: 4]')
    parser.add_argument('--length', type=int, default=120,
                        help='Number of time points in each sample [default: 120]')
    # condition sampling on a particular images
    parser.add_argument('--condition-on-conditions', type=int, default=None,
                        help='If True, generate text conditioned on an image.')
    # condition sampling on a particular text
    parser.add_argument('--condition-on-targets', type=int, default=None, 
                        help='If True, generate images conditioned on a text.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    X_train = np.load('/home/MihailBogojeski/git/multimodaldiffeq/data/basf-iap/synthetic/X_train.npy')
    Y_train = np.load('/home/MihailBogojeski/git/multimodaldiffeq/data/basf-iap/synthetic/Y_train.npy')
    Y_train = Y_train[:, 0:1]

    condition_mean = np.mean(X_train, axis=0)
    condition_std = np.std(X_train, axis=0)
    target_mean = np.mean(Y_train, axis=0)
    target_std = np.std(Y_train, axis=0)

    condition_mean = torch.Tensor(condition_mean)
    target_mean = torch.Tensor(target_mean)
    condition_std = torch.Tensor(condition_std)
    target_std = torch.Tensor(target_std)

    scalers = [StandardNormalization(condition_mean, condition_std),
               StandardNormalization(target_mean, target_std)]

    model = load_checkpoint(args.model_path, SeqMVAE, use_cuda=args.cuda,
                            keys=['encoders', 'decoders', 'x_maps', 'z_map',
                                  'prior', 'recurrents'])
    model.eval()
    if args.cuda:
        model.cuda()
        for scaler in scalers:
            scaler.cuda()

    sample_list = []
    condition_list = []
    target_list = []
    for i in range(args.n_samples):
        model.init_states(1)

        # mode 1: unconditional generation
        if not args.condition_on_conditions and not args.condition_on_targets:
            total_length = args.length
            input_length = args.length
            inputs = [None, None]
        # mode 2: generate conditioned on image
        elif args.condition_on_conditions and not args.condition_on_targets:
            condition, _, length = fetch_ts_sample()
            condition_list.append(condition.numpy())
            if args.cuda:
                condition = condition.cuda()
            print('condition shape', condition.shape)
            print('length', length)
            input_length = int(length.squeeze())
            total_length = condition.shape[0]
            inputs = [condition.unsqueeze(0), None]
        # mode 3: generate conditioned on text
        elif args.condition_on_targets and not args.condition_on_conditions:
            _, target, length = fetch_ts_sample()
            target_list.append(target.numpy())
            if args.cuda:
                target = target.cuda()
            input_length = int(length.squeeze())
            total_length = target.shape[0]
            inputs = [None, target.unsqueeze(0)]
        elif args.condition_on_conditions and args.condition_on_targets:
            condition, target, length = fetch_ts_sample()
            condition_list.append(condition.numpy())
            target_list.append(target.numpy())
            if args.cuda:
                condition = condition.cuda()
                target = target.cuda()
            input_length = int(length.squeeze())
            total_length = target.shape[0]
            inputs = [condition.unsqueeze(0), target.unsqueeze(0)]
        # sample from uniform gaussian
        print(inputs)
        seq_inputs = transform_to_sequence(inputs, total_length)

        recon_condition = None
        recon_target = None
        for j in range(input_length):
            recs, mu, logvar, p_mu, p_var, logj, z0, zk = model(seq_inputs[j])
            if recon_condition is None:
                recon_condition = recs[0]
                recon_target = recs[1]
            else:
                recon_condition = torch.cat((recon_condition, recs[0]), dim=0)
                recon_target = torch.cat((recon_target, recs[1]), dim=0)

        recon_condition = recon_condition.cpu().detach().numpy()
        recon_target = recon_target.cpu().detach().numpy()
        print('condition size', recon_condition.shape)
        print('target size', recon_target.shape)
        sample_list.append(np.concatenate((recon_condition, recon_target), axis=1))

        # save image samples to filesystem

    np.save('ts_sample.npy', sample_list)
    if args.condition_on_targets:
        np.save('ts_target.npy', target_list)
    if args.condition_on_conditions:
        np.save('ts_cond.npy', condition_list)
