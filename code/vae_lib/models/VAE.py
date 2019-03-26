#VAE baseclass for CNFVAE
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import code.vae_lib.models.flows as flows
from code.vae_lib.models.layers import GatedConv2d, GatedConvTranspose2d


class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type

        if self.input_size == [1, 28, 28] or self.input_size == [3, 28, 28]:
            self.last_kernel_size = 7
        elif self.input_size == [1, 28, 20]:
            self.last_kernel_size = (7, 5)
        else:
            raise ValueError('invalid input size!!')

        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()

        self.q_z_nn_output_dim = 256

        # auxiliary
        if args.cuda:
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """

        if self.input_type == 'binary':
            q_z_nn = nn.Sequential(
                GatedConv2d(self.input_size[0], 32, 5, 1, 2),
                GatedConv2d(32, 32, 5, 2, 2),
                GatedConv2d(32, 64, 5, 1, 2),
                GatedConv2d(64, 64, 5, 2, 2),
                GatedConv2d(64, 64, 5, 1, 2),
                GatedConv2d(64, 256, self.last_kernel_size, 1, 0),
            )
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(256, self.z_size),
                nn.Softplus(),
            )
            return q_z_nn, q_z_mean, q_z_var

        elif self.input_type == 'multinomial':
            act = None

            q_z_nn = nn.Sequential(
                GatedConv2d(self.input_size[0], 32, 5, 1, 2, activation=act),
                GatedConv2d(32, 32, 5, 2, 2, activation=act),
                GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, 2, activation=act),
                GatedConv2d(64, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
            )
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(nn.Linear(256, self.z_size), nn.Softplus(), nn.Hardtanh(min_val=0.01, max_val=7.))
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """

        num_classes = 256

        if self.input_type == 'binary':
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0),
                GatedConvTranspose2d(64, 64, 5, 1, 2),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                GatedConvTranspose2d(32, 32, 5, 1, 2),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1), GatedConvTranspose2d(32, 32, 5, 1, 2)
            )

            p_x_mean = nn.Sequential(nn.Conv2d(32, self.input_size[0], 1, 1, 0), nn.Sigmoid())
            return p_x_nn, p_x_mean

        elif self.input_type == 'multinomial':
            act = None
            p_x_nn = nn.Sequential(
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
            )

            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.input_size[0] * num_classes, 1, 1, 0),
                # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
            )

            return p_x_nn, p_x_mean

        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        #print("var", var)
        std = var.sqrt()
        #print(std)
        eps = self.FloatTensor(std.size()).normal_()
        z = eps.mul(std).add_(mu)
        #print (z)
        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """

        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)

        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """

        # mean and variance of z
        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z



class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)
        self.experts = ProductOfExperts()
        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.
        self.h_size = args.made_h_size

        self.h_context = nn.Linear(self.q_z_nn_output_dim, self.h_size)

        # Flow parameters
        self.num_flows = args.num_flows
        self.flow = flows.IAF(
            z_size=self.z_size, num_flows=self.num_flows, num_hidden=1, h_size=self.h_size, conv2d=False
        )

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        """

        h, mean_z, var_z = self.infer(x, x)
        h = h.view(-1, self.q_z_nn_output_dim)
        #mean_z = self.q_z_mean(h)
        #var_z = self.q_z_var(h)
        h_context = self.h_context(h)

        return mean_z, var_z, h_context

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
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)

        z_0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        z_k, self.log_det_j = self.flow(z_0, h_context)
        #print("PRINT",z_k, self.log_det_j)
        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k


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

