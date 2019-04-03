# !/usr/bin/env python
# -*- coding: utf-8 -*-
#train VAE with flows
from __future__ import print_function

import torchvision

from code.vae_lib.models.train_misc import count_nfe, override_divergence_fn


import argparse
import sys
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import random
from code.vae_lib.optimization.loss import calculate_loss, elbo_loss
import os

from torch.autograd import Variable
from torchvision import transforms
import datetime
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, FakeData, CIFAR10
import lib.utils as utils
import lib.layers.odefunc as odefunc
import lib.layers as layers

import code.vae_lib.models.VAE as VAE
import code.vae_lib.models.CNFVAE as CNFVAE
import code.vae_lib.models.fashionmnistmodel as FCNFVAE
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
    '-es', '--early_stopping_epochs', type=int, default=35, metavar='EARLY_STOPPING',
    help='number of early stopping epochs'
)

parser.add_argument(
    '-bs', '--batch_size', type=int, default=2048, metavar='BATCH_SIZE', help='input batch size for training'
)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, metavar='LEARNING_RATE', help='learning rate')

parser.add_argument(
    '-w', '--warmup', type=int, default=10, metavar='N',
    help='number of epochs for warm-up. Set to 0 to turn warmup off.'
)
parser.add_argument('--max_beta', type=float, default=1., metavar='MB', help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB', help='min beta for warm-up')
parser.add_argument(
    '-f', '--flow', type=str, default='cnf', choices=[
        'planar', 'iaf', 'householder', 'orthogonal', 'triangular', 'cnf', 'cnf_bias', 'cnf_hyper', 'cnf_rank',
        'cnf_lyper', 'no_flow'
    ], help="""Type of flows to use, no flows can also be selected"""
)
parser.add_argument('-r', '--rank', type=int, default=1)
parser.add_argument(
    '-nf', '--num_flows', type=int, default=4, metavar='NUM_FLOWS',
    help='Number of flow layers, ignored in absence of flows'
)
parser.add_argument(
    '-nv', '--num_ortho_vecs', type=int, default=8, metavar='NUM_ORTHO_VECS',
    help=""" For orthogonal flow: How orthogonal vectors per flow do you need.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-nh', '--num_householder', type=int, default=8, metavar='NUM_HOUSEHOLDERS',
    help=""" For Householder Sylvester flow: Number of Householder matrices per flow.
                    Ignored for other flow types."""
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
    "--layer_type", type=str, default="blend",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='512-512')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=False)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS)
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
parser.add_argument('--annealing-epochs', type=int, default=40, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

seed = 20
args.manual_seed = seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

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

    if args.flow != 'no_flow':
        snap_dir += '_' + 'num_flows_' + str(args.num_flows)

    if args.flow == 'orthogonal':
        snap_dir = snap_dir + '_num_vectors_' + str(args.num_ortho_vecs)
    elif args.flow == 'orthogonalH':
        snap_dir = snap_dir + '_num_householder_' + str(args.num_householder)
    elif args.flow == 'iaf':
        snap_dir = snap_dir + '_madehsize_' + str(args.made_h_size)

    elif args.flow == 'permutation':
        snap_dir = snap_dir + '_' + 'kernelsize_' + str(args.kernel_size)
    elif args.flow == 'mixed':
        snap_dir = snap_dir + '_' + 'num_householder_' + str(args.num_householder)
    elif args.flow == 'cnf_rank':
        snap_dir = snap_dir + '_rank_' + str(args.rank) + '_' + args.dims + '_num_blocks_' + str(args.num_blocks)
    elif 'cnf' in args.flow:
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
    #train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
    args.dynamic_binarization = False
    args.input_type = 'binary'
    transform = transforms.Compose(
        [transforms.Grayscale(1),
        transforms.Resize((28, 28), interpolation=2),
        transforms.ToTensor()
         #transforms.Normalize((0.5,), (0.5,))
         ])
    args.input_size = [1, 28, 28]
    train_loader = torch.utils.data.DataLoader(
        FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True)
    N_mini_batches = len(train_loader)
    test_loader = torch.utils.data.DataLoader(
        FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)

    if not args.evaluate:

        # ==============================================================================================================
        # SELECT MODEL
        # ==============================================================================================================
        # flow parameters and architecture choice are passed on to model through args

        if args.flow == 'no_flow':
            model = VAE.VAE(args)
        elif args.flow == 'planar':
            model = VAE.PlanarVAE(args)
        elif args.flow == 'iaf':
            model = VAE.IAFVAE(args)
        elif args.flow == 'orthogonal':
            model = VAE.OrthogonalSylvesterVAE(args)
        elif args.flow == 'householder':
            model = VAE.HouseholderSylvesterVAE(args)
        elif args.flow == 'triangular':
            model = VAE.TriangularSylvesterVAE(args)
        elif args.flow == 'cnf':
            model = CNFVAE.CNFVAE(args)
        elif args.flow == 'cnf_bias':
            model = CNFVAE.AmortizedBiasCNFVAE(args)
        elif args.flow == 'cnf_hyper':
            model = CNFVAE.HypernetCNFVAE(args)
        elif args.flow == 'cnf_lyper':
            model = CNFVAE.LypernetCNFVAE(args)
        elif args.flow == 'cnf_rank':
            model = CNFVAE.AmortizedLowRankCNFVAE(args)
        else:
            raise ValueError('Invalid flow choice')

        if args.retrain_encoder:
            logger.info(f"Initializing decoder from {args.model_path}")
            dec_model = torch.load(args.model_path)
            dec_sd = {}
            for k, v in dec_model.state_dict().items():
                if 'p_x' in k:
                    dec_sd[k] = v
            model.load_state_dict(dec_sd, strict=False)

        if args.cuda:
            logger.info("Model on GPU")
            model.cuda()

        logger.info(model)

        if args.retrain_encoder:
            parameters = []
            logger.info('Optimizing over:')
            for name, param in model.named_parameters():
                if 'p_x' not in name:
                    logger.info(name)
                    parameters.append(param)
        else:
            parameters = model.parameters()

        #optimizer = optim.Adamax(parameters, lr=args.learning_rate, eps=1.e-7)
        optimizer = optim.Adamax(parameters, args.learning_rate, eps=1.e-7)
        # ==================================================================================================================
        # TRAINING AND EVALUATION
        # ==================================================================================================================
        def train(epoch):
            override_divergence_fn(model, "approximate")
            beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
            model.train()
            train_loss_meter = AverageMeter()
            # NOTE: is_paired is 1 if the example is paired
            for batch_idx, (image, text) in enumerate(train_loader):

                if epoch < args.annealing_epochs:
                    # compute the KL annealing factor for the current mini-batch in the current epoch
                    annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                        float(args.annealing_epochs * N_mini_batches))
                else:
                    # by default the KL annealing factor is unity
                    annealing_factor = 1.0

                if args.cuda:
                    image = image.cuda()
                    text = text.cuda()

                image = Variable(image)
                text = Variable(text)

                batch_size = len(image)

                # refresh the optimizer
                optimizer.zero_grad()
                # pass data through model
                recon_image_1, recon_text_1, mu_1, logvar_1, logj1, z01, zk1 = model(image, text)
                recon_image_2, recon_text_2, mu_2, logvar_2, logj2, z02, zk2 = model(image)
                recon_image_3, recon_text_3, mu_3, logvar_3, logj3, z03, zk3 = model(text=text)

                # compute ELBO for each data combo
                joint_loss, rec1_1, rec1_2, kl_1 = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1, z01, zk1, logj1,
                                           args, lambda_image=1.0, lambda_text=10.0, annealing_factor=annealing_factor, beta=beta)
                image_loss, rec1_2, rec2_2, kl_2= elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, z02, zk2, logj2,
                                                      args, lambda_image=1.0, lambda_text=10.0, annealing_factor=annealing_factor, beta=beta)
                text_loss, rec1, rec2, kl = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3, z03, zk3, logj3,
                                                     args, lambda_image=1.0, lambda_text=10.0, annealing_factor=annealing_factor, beta=beta)
                #print("TEXT", r, "TEXTLOSS",text_loss, image_loss.shape, image_loss)
                train_loss = joint_loss + image_loss + text_loss # joint_loss# ovie se tie 3 losses, za sekoja kombinacija poedinecno ama aj so 2 ke testiram
                train_loss_meter.update(train_loss.item(), batch_size)
                # compute gradients and take step
                train_loss.backward()
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                        epoch, batch_idx * len(image), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))

            print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))

        def test(epoch):

            model.eval()
            beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
            image_loss_meter = AverageMeter()
            text_loss_meter = AverageMeter()
            test_loss_meter = AverageMeter()
            override_divergence_fn(model, "brute_force")
            for batch_idx, (image, text) in enumerate(test_loader):
                if args.cuda:
                    image = image.cuda()
                    text = text.cuda()
                image = Variable(image, volatile=True)
                text = Variable(text, volatile=True)
                batch_size = len(image)

                recon_image_1, recon_text_1, mu_1, logvar_1, logj1, z01, zk1 = model(image, text)
                recon_image_2, recon_text_2, mu_2, logvar_2, logj2, z02, zk2 = model(image)
                recon_image_3, recon_text_3, mu_3, logvar_3, logj3, z03, zk3 = model(text=text)

                # compute ELBO for each data combo
                joint_loss, rec1, rec2, kl = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1, z01, zk1, logj1,args)
                image_loss_meter.update(rec1.mean().item(), batch_size)
                text_loss_meter.update(rec2.mean().item(), batch_size)
                image_loss, rec1, rec2, kl = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, z02, zk2, logj2, args)
                image_loss_meter.update(rec1.mean().item(), batch_size)

                text_loss, rec1, rec2, kl = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3, z03, zk3, logj3,args)
                text_loss_meter.update(rec2.mean().item(), batch_size)

                test_loss = joint_loss + image_loss + text_loss
                test_loss_meter.update(test_loss.item(), batch_size)


            print('====> Test image loss: {:.4f}'.format(image_loss_meter.avg))
            print('====> Test text loss: {:.4f}'.format(text_loss_meter.avg))
            print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
            return test_loss_meter.avg

        best_loss = sys.maxsize
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            #print ("Test")
            test_loss = test(epoch)
            is_best   = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            # save the best model and current model
            save_checkpoint({
                'state_dict': model.state_dict(),
                'args':args,
                'best_loss': best_loss,
                'n_latents': args.z_size,
                'optimizer' : optimizer.state_dict(),
            }, is_best, folder='./trained_models')


if __name__ == "__main__":

    run(args, kwargs)
