import torch
import torch.nn as nn
from torch.autograd import Variable
from code.vae_lib.models.train_misc import build_model_tabular
from .VAE import VAE
from code.vae_lib.models.model import ProductOfExperts, prior_expert


def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.z_size,)


def concat_layer_num_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim + out_dim


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GenMVAE(VAE):

    def __init__(self, args, encoders, decoders):
        super(GenMVAE, self).__init__(args)
        self.experts = ProductOfExperts()
        # CNF model
        self.encoders = nn.ModuleList()
        for enc in encoders:
            self.encoders.append(enc)
        self.decoders = nn.ModuleList()
        for dec in decoders:
            self.decoders.append(dec)
        self.z_size = args.z_size
        if args.cuda:
            self.cuda()

    def forward(self, inputs, lengths=None):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var = self.encode(inputs, lengths)
        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)
        # z0 = z0.to(z_mu)
        reconstructions = []

        for dec in self.decoders:
            if lengths is None:
                reconstructions.append(dec(z0))
            else:
                reconstructions.append(dec(z0, length=max(lengths)))
        print('z0 shape', z0.shape)

        return reconstructions, z_mu, z_var, torch.zeros((z0.shape[0], )).to(z0), z0, None

    def encode(self, inputs, lengths=None):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        mean_z, var_z = self.infer(inputs, lengths)

        return mean_z, var_z

    def infer(self, inputs, lengths=None):

        batch_size = 1
        for inp in inputs:
            if inp is not None:
                batch_size = inp.size(0)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.z_size),
                                  use_cuda=use_cuda)

        for inp, enc in zip(inputs, self.encoders):
            if inp is None:
                continue
            if lengths is None:
                mean_z, var_z = enc(inp)
            else:
                mean_z, var_z = enc(inp, lengths)
            mu = torch.cat((mu, mean_z.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, var_z.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians

        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

