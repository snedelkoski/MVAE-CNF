# !/usr/bin/env python
# -*- coding: utf-8 -*-
# train VAE with flows
from __future__ import print_function

import argparse
import sys
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from code.vae_lib.optimization.loss import vaegan_losses, cross_entropy, binary_cross_entropy_with_logits
import os

from code.vae_lib.models.model import TextReprEncoder, TextDiscriminator, TextDecoder, ImageEncoder, ImageDiscriminator, ImageDecoder

from torch.autograd import Variable
from torchvision import transforms
import datetime
from torchvision.datasets import MNIST
import lib.utils as utils
import lib.layers.odefunc as odefunc

# import code.vae_lib.models.CNFVAE as CNFVAE
from code.vae_lib.models.MVAEGAN import MCVAEGAN
from code.vae_lib.models.MVAE import GenMVAE
from code.vae_lib.models.train import AverageMeter, save_checkpoint
from code.vae_lib.optimization.training import train, evaluate
from code.vae_lib.utils.load_data import load_dataset
from code.vae_lib.utils.plotting import plot_training_curve
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser(description='PyTorch Sylvester Normalizing flows')

parser.add_argument(
    '-d', '--dataset', type=str, default='mnist', choices=['mnist', 'freyfaces', 'omniglot', 'caltech'],
    metavar='DATASET', help='Dataset choice.'
)

parser.add_argument(
    '-freys', '--freyseed', type=int, default=123, metavar='FREYSEED',
    help="""Seed for shuffling frey face dataset for test split. Ignored for other datasets.
                    Results in paper are produced with seeds 123, 321, 231"""
)

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--manual_seed', default=10, type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument(
    '-li', '--log_interval', type=int, default=1, metavar='LOG_INTERVAL',
    help='how many batches to wait before logging training status'
)

parser.add_argument(
    '-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
    help='output directory for model snapshots etc.'
)

# optimization settings
parser.add_argument(
    '-e', '--epochs', type=int, default=100, metavar='EPOCHS', help='number of epochs to train (default: 2000)'
)
parser.add_argument(
    '--num_inputs', type=int, default=2, metavar='EPOCHS', help='number of epochs to train (default: 2000)'
)
parser.add_argument(
    '--num_subsets', type=int, default=0, metavar='EPOCHS', help='number of epochs to train (default: 2000)'
)
parser.add_argument(
    '-es', '--early_stopping_epochs', type=int, default=35, metavar='EARLY_STOPPING',
    help='number of early stopping epochs'
)

parser.add_argument(
    '-bs', '--batch_size', type=int, default=2048, metavar='BATCH_SIZE', help='input batch size for training'
)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, metavar='LEARNING_RATE', help='learning rate')

parser.add_argument(
    '-w', '--warmup', type=int, default=10, metavar='N',
    help='number of epochs for warm-up. Set to 0 to turn warmup off.'
)
parser.add_argument('--max_beta', type=float, default=1., metavar='MB', help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB', help='min beta for warm-up')
parser.add_argument('-r', '--rank', type=int, default=1)
parser.add_argument(
    '-nf', '--num_flows', type=int, default=4, metavar='NUM_FLOWS',
    help='Number of flow layers, ignored in absence of flows'
)
parser.add_argument(
    '-mhs', '--made_h_size', type=int, default=320, metavar='MADEHSIZE',
    help='Width of mades for iaf. Ignored for all other flows.'
)
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE', help='how many stochastic hidden units')
# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

# CNF settings
parser.add_argument(
    "--layer_type", type=str, default="concat",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='512-512')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=False)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--flow", type=str, default="cnf")
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='midpoint', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)
# evaluation
parser.add_argument('--evaluate', type=eval, default=False, choices=[True, False])
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--retrain_encoder', type=eval, default=False, choices=[True, False])
parser.add_argument('--annealing-epochs', type=int, default=1, metavar='N',
                    help='number of epochs to anneal KL for [default: 200]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.manual_seed(0)
np.random.seed(args.manual_seed)

if args.cuda:
    # gpu device number
    torch.cuda.set_device(args.gpu_num)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):
    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = os.path.join(args.out_dir, 'vae_' + args.dataset + '_')
    snap_dir = snapshots_path + args.flow

    snap_dir = snap_dir + '_' + args.dims + '_num_blocks_' + str(args.num_blocks)

    if args.retrain_encoder:
        snap_dir = snap_dir + '_retrain-encoder_'
    elif args.evaluate:
        snap_dir = snap_dir + '_evaluate_'

    snap_dir = snap_dir + '__' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    # logger
    utils.makedirs(args.snap_dir)

    logger = utils.get_logger(logpath=os.path.join(args.snap_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # SAVING
    torch.save(args, snap_dir + args.flow + '.config')

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    # train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
    args.dynamic_binarization = False
    args.input_type = 'binary'

    args.input_size = [1, 28, 28]
    train_loader = torch.utils.data.DataLoader(
        MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    N_mini_batches = len(train_loader)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)

    if not args.evaluate:

        # ==============================================================================================================
        # SELECT MODEL
        # ==============================================================================================================
        # flow parameters and architecture choice are passed on to model through args
        encoders = [ImageEncoder, TextReprEncoder]
        decoders = [ImageDecoder, TextDecoder]
        embeddings = [nn.Linear(10, 512)]

        loss_funcs = [F.mse_loss, F.mse_loss]

        # model = GenMVAE(args, encoders, decoders)
        model = MCVAEGAN(args, encoders, decoders, discriminators)

        # if args.retrain_encoder:
        #     logger.info(f"Initializing decoder from {args.model_path}")
        #     dec_model = torch.load(args.model_path)
        #     dec_sd = {}
        #     for k, v in dec_model.state_dict().items():
        #         if 'p_x' in k:
        #             dec_sd[k] = v
        #     model.load_state_dict(dec_sd, strict=False)

        if args.cuda:
            logger.info("Model on GPU")
            model.cuda()

        logger.info(model)

        # if args.retrain_encoder:
        #     parameters = []
        #     logger.info('Optimizing over:')
        #     for name, param in model.named_parameters():
        #         if 'p_x' not in name:
        #             logger.info(name)
        #             parameters.append(param)
        # else:
        parameters = model.parameters()

        # optimizer = optim.Adamax(parameters, lr=args.learning_rate, eps=1.e-7)
        optimizer = optim.Adam(parameters, args.learning_rate)

        # ==================================================================================================================
        # TRAINING AND EVALUATION
        # ==================================================================================================================
        def train(epoch):
            model.train()
            enc_loss_meter = AverageMeter()
            dec_loss_meter = AverageMeter()
            dis_loss_meter = AverageMeter()
            binary_selections = [np.ones((args.num_inputs, 1))]

            for i in range(args.num_inputs):
                selection = np.zeros((args.num_inputs, 1))
                selection[i] = 1
                binary_selections.append(selection)

            for i in range(args.num_subsets):
                selection = np.random.choice([0, 1], size=args.num_inputs)
                binary_selections.append(selection)

            # override_divergence_fn(model, "brute_force")
            # NOTE: is_paired is 1 if the example is paired
            for batch_idx, (image, text) in enumerate(train_loader):
                if epoch < args.annealing_epochs:
                    # compute the KL annealing factor for the current mini-batch in the current epoch
                    annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                        float(args.annealing_epochs * N_mini_batches))
                else:
                    # by default the KL annealing factor is unity
                    annealing_factor = 1.0

                text_repr = torch.zeros((text.shape[0], 10))

                for i in range(text.shape[0]):
                    text_repr[i, text[i]] = 1

                text_repr += 0.001

                if args.cuda:
                    image = image.cuda()
                    text_repr = text_repr.cuda()

                batch_size = len(image)

                # refresh the optimizer
                optimizer.zero_grad()

                # pass data through model
                enc_loss = 0
                dec_loss = 0
                dis_loss = 0
                inputs = [image, text_repr]
                # compute ELBO for each data combo
                for sel in binary_selections:
                    sel_inputs = [inp if flag else None for flag, inp in zip(sel, inputs)]

                    recs, mu, logvar, logj, z0, zk = model(sel_inputs)

                    z_aux = torch.randn(z0.shape).to(z0)
                    recs_aux = model.decode(z_aux)

                    preds_true, feats_true = model.discriminate_with_features(sel_inputs)
                    preds_fake, feats_fake = model.discriminate_with_features(recs)
                    preds_aux = model.discriminate(recs_aux)

                    GAN_loss, recon_loss, kl = vaegan_losses(feats_fake, feats_true, preds_true, preds_fake,
                                                             preds_aux, loss_funcs, mu, logvar, z0, zk, logj,
                                                             args, annealing_factor=annealing_factor)
                    enc_loss += kl + recon_loss

                    dec_loss += recon_loss - GAN_loss

                    dis_loss += GAN_loss

                enc_loss_meter.update(enc_loss.item(), batch_size)
                dec_loss_meter.update(dec_loss.item(), batch_size)
                dis_loss_meter.update(dis_loss.item(), batch_size)

                # compute gradients and take step
                model.freeze_params(True, False, False)
                enc_loss.backward(retain_graph=True)
                # print('encoders grad:')
                # for enc in model.encoders:
                #     for name, p in enc.named_parameters():
                #         print(name, p.grad.mean())
                #         # break
                #     break

                model.freeze_params(False, True, False)
                dec_loss.backward(retain_graph=True)
                # print('decoders grad:')
                # for dec in model.decoders:
                #     for name, p in dec.named_parameters():
                #         print(name, p.grad.mean())
                #         # break
                #     break

                model.freeze_params(False, False, True)
                dis_loss.backward()
                # print('discriminators grad:')
                # for dis in model.discriminators:
                #     for name, p in dis.named_parameters():
                #         print(name, p.grad.mean())
                #         # break
                #     break
                model.freeze_params(True, True, True)
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    print(('Train Epoch: {} [{}/{} ({:.0f}%)]\tEncoder Loss: {:.3f}\tDecoder Loss: {:.3f}' +
                          '\tDiscriminator Loss: {:.3f}\tAnnealing-Factor: {:.3f}').format(
                              epoch, batch_idx * len(image), len(train_loader.dataset),
                              100. * batch_idx / len(train_loader), enc_loss_meter.avg, dec_loss_meter.avg, dis_loss_meter.avg, annealing_factor))

            print(('====> Epoch: {}\tEncoder Loss: {:.3f}\tDecoder Loss: {:.3f}' +
                  '\tDiscriminator Loss: {:.3f}').format(epoch, enc_loss_meter.avg, dec_loss_meter.avg, dis_loss_meter.avg))

        def test(epoch):

            model.eval()
            # beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
            enc_loss_meter = AverageMeter()
            dec_loss_meter = AverageMeter()
            dis_loss_meter = AverageMeter()
            binary_selections = [np.ones((args.num_inputs, 1))]

            for i in range(args.num_inputs):
                selection = np.zeros((args.num_inputs, 1))
                selection[i] = 1
                binary_selections.append(selection)

            for i in range(args.num_subsets):
                selection = np.random.choice([0, 1], size=args.num_inputs)
                binary_selections.append(selection)

            # override_divergence_fn(model, "brute_force")
            for batch_idx, (image, text) in enumerate(test_loader):

                text_repr = torch.zeros((text.shape[0], 10))

                for i in range(text.shape[0]):
                    text_repr[i, text[i]] = 1

                text_repr += 0.001

                if args.cuda:
                    image = image.cuda()
                    text_repr = text_repr.cuda()

                # image = Variable(image, volatile=True)
                # text_repr = Variable(text_repr, volatile=True)
                batch_size = len(image)

                enc_loss = 0
                dec_loss = 0
                dis_loss = 0
                inputs = [image, text_repr]
                # compute ELBO for each data combo
                for sel in binary_selections:
                    sel_inputs = [inp if flag else None for flag, inp in zip(sel, inputs)]
                    recs, mu, logvar, logj, z0, zk = model(sel_inputs)

                    z_aux = torch.randn(z0.shape).to(z0)

                    recs_aux = model.decode(z_aux)

                    preds_true, feats_true = model.discriminate_with_features(sel_inputs)
                    preds_fake, feats_fake = model.discriminate_with_features(recs)
                    preds_aux = model.discriminate(recs_aux)

                    GAN_loss, recon_loss, kl = vaegan_losses(feats_fake, feats_true, preds_true, preds_fake, preds_aux,
                                                             loss_funcs, mu, logvar, z0, zk, logj, args)
                    enc_loss += kl + recon_loss

                    dec_loss += recon_loss - GAN_loss

                    dis_loss += GAN_loss

                enc_loss_meter.update(enc_loss.item(), batch_size)
                dec_loss_meter.update(dec_loss.item(), batch_size)
                dis_loss_meter.update(dis_loss.item(), batch_size)

            print(('====> Encoder Loss: {:.3f}\tDecoder Loss: {:.3f}' +
                  '\tDiscriminator Loss: {:.3f}').format(enc_loss_meter.avg, dec_loss_meter.avg, dis_loss_meter.avg))
            return enc_loss_meter.avg, dec_loss_meter.avg, dis_loss_meter.avg

        best_loss = sys.maxsize
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            # print ("Test")
            test_loss, _, _ = test(epoch)
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            # save the best model and current model
            save_checkpoint({
                'state_dict': model.state_dict(),
                'args': args,
                'encoders': encoders,
                'decoders': decoders,
                'discriminators': discriminators,
                'best_loss': best_loss,
                'n_latents': args.z_size,
                'optimizer': optimizer.state_dict(),
            }, is_best, folder='./trained_models')


if __name__ == "__main__":

    run(args, kwargs)
