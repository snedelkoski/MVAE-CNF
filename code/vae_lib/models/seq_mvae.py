import torch
import torch.nn as nn
from torch.autograd import Variable
from code.vae_lib.models.train_misc import build_model_tabular
from .VAE import VAE
from code.vae_lib.models.model import ProductOfExperts, prior_expert
from code.vae_lib.models.MVAE import GenMVAE


def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.z_size,)


def concat_layer_num_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim + out_dim


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SeqMVAE(GenMVAE):

    def __init__(self, args, encoders, decoders, x_maps, z_map, prior, recurrents):
        super().__init__(args, [], [])

        self.input_feats = None
        self.z_feats = None

        self.encoders = nn.ModuleList()
        for enc in encoders:
            self.encoders.append(enc)

        self.decoders = nn.ModuleList()
        for dec in decoders:
            self.decoders.append(dec)

        self.x_maps = nn.ModuleList()
        for x_map in x_maps:
            self.x_maps.append(x_map)

        self.prior = prior

        self.z_map = z_map

        self.recurrents = nn.ModuleList()
        for rec in recurrents:
            if args.cuda:
                rec.device = 'cuda:0'
            self.recurrents.append(rec)

        if args.cuda:
            self.cuda()

    def forward(self, inputs):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var, p_mu, p_var = self.encode(inputs)
        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)
        # z0 = z0.to(z_mu)
        reconstructions, x_feats, z_feats = self.decode(z0)

        self.update_hidden(x_feats, z_feats)

        return reconstructions, z_mu, z_var, p_mu, p_var, torch.zeros((z0.shape[0], )).to(z0), z0, None

    def encode(self, inputs):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        mean_z, var_z, p_mean, p_var = self.infer(inputs)

        return mean_z, var_z, p_mean, p_var

    def get_prior(self):
        hidden_sum = torch.zeros(self.recurrents[0].hidden.shape).to(self.recurrents[0].hidden)

        for rec in self.recurrents:
            hidden_sum += rec.hidden

        mu, logvar = self.prior(hidden_sum)

        return mu.unsqueeze(0), logvar.unsqueeze(0)

    def infer(self, inputs):

        # initialize the universal prior expert

        mu, logvar = self.get_prior()
        prior_mu = mu.squeeze()
        prior_logvar = logvar.squeeze()
        for inp, enc, x_map, rec in zip(inputs, self.encoders, self.x_maps,
                                        self.recurrents):
            if inp is None:
                continue
            inp_map = x_map(inp)
            mean_z, var_z = enc([inp_map, rec.hidden])
            mu = torch.cat((mu, mean_z.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, var_z.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians

        mu, logvar = self.experts(mu, logvar)

        return mu, logvar, prior_mu, prior_logvar

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

    def decode(self, z0):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        z_feats = self.z_map(z0)
        recons = []
        for dec, rec in zip(self.decoders, self.recurrents):
            recons.append(dec([z_feats, rec.hidden]))

        x_feats = []
        for recon, x_map in zip(recons, self.x_maps):
            x_feats.append(x_map(recon))

        return recons, x_feats, z_feats

    def update_hidden(self, x_feats, z_feats):
        for rec, x_feat in zip(self.recurrents, x_feats):
            rec([x_feat, z_feats])

    def init_states(self, batch_size):
        for rec in self.recurrents:
            rec.init_states(batch_size)
