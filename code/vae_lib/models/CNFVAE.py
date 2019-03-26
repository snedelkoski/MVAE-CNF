#this is from FFJORD

import torch
import torch.nn as nn
from torch.autograd import Variable

from code.vae_lib.models.train_misc import build_model_tabular
import lib.layers as layers
from .VAE import VAE
import lib.layers.diffeq_layers as diffeq_layers
from code.lib.layers.odefunc import NONLINEARITIES

from torchdiffeq import odeint_adjoint as odeint


def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.z_size,)


def concat_layer_num_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim + out_dim


class CNFVAE(VAE):

    def __init__(self, args):
        super(CNFVAE, self).__init__(args)
        print ("CNF created")
        self.experts = ProductOfExperts()
        # CNF model
        self.cnf = build_model_tabular(args, args.z_size)

        if args.cuda:
            self.cuda()

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        h, mean_z, var_z = self.infer(x, x)
        #h = h.view(-1, self.q_z_nn_output_dim)

        return mean_z, var_z
    def infer(self, image1=None, image2=None):
        batch_size = image1.size(0) if image1 is not None else image2.size(0)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.z_size),
                                  use_cuda=use_cuda)

        if image1 is not None:
            h = self.q_z_nn(image1)
            h = h.view(-1, self.q_z_nn_output_dim)
            mean_z = self.q_z_mean(h)
            var_z = self.q_z_var(h)
            mu = torch.cat((mu, mean_z.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, var_z.unsqueeze(0)), dim=0)

        if image2 is not None:
            h = self.q_z_nn(image2)
            h = h.view(-1, self.q_z_nn_output_dim)
            mean_z = self.q_z_mean(h)
            var_z = self.q_z_var(h)
            mu = torch.cat((mu, mean_z.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, var_z.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        #TODO: solve h
        #print ("logvar", logvar)
        mu, logvar = self.experts(mu, logvar)
        #print("A:logvar", logvar)
        return h, mu, logvar
    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        z_mu, z_var = self.encode(x)

        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)

        zero = torch.zeros(x.shape[0], 1).to(x)
        zk, delta_logp = self.cnf(z0, zero)  # run model forward

        x_mean = self.decode(zk)

        return x_mean, z_mu, z_var, -delta_logp.view(-1), z0, zk


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        #print ("logvar", logvar)
        var       = torch.exp(logvar) + eps
        #print ("2:", var)
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        #print("3:", T)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        #print ("4", pd_mu)
        pd_var    = 1. / torch.sum(T, dim=0)
        #print ("5", pd_var)
        #changed
        pd_logvar = torch.log(1+pd_var + eps)
        #print ("pd", pd_logvar)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

