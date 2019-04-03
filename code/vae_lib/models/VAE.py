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

        #self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        #self.p_x_nn, self.p_x_mean = self.create_decoder()

        self.q_z_nn_output_dim = 256

        # auxiliary
        #if args.cuda:
        #    self.FloatTensor = torch.cuda.FloatTensor
        #else:
        #    self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        #self.log_det_j = self.FloatTensor(1).zero_()

class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)

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

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        h_context = self.h_context(h)

        return mean_z, var_z, h_context

    def forward(self, x):
        """
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        # mean and variance of z
        z_mu, z_var, h_context = self.encode(x)
        # sample z
        z_0 = self.reparameterize(z_mu, z_var)

        # iaf flows
        z_k, self.log_det_j = self.flow(z_0, h_context)

        # decode
        x_mean = self.decode(z_k)

        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k

