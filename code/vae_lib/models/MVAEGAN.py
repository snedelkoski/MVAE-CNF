import torch
import torch.nn as nn
from torch.autograd import Variable
from code.vae_lib.models.train_misc import build_model_tabular
from .VAE import VAE
from code.vae_lib.models.model import ProductOfExperts, prior_expert, MultimodalDiscriminator



def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.z_size,)


def concat_layer_num_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim + out_dim


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class MVAEGAN(VAE):

    def __init__(self, args, encoders, decoders, discriminators):
        super(MVAEGAN, self).__init__(args)
        self.experts = ProductOfExperts()
        # CNF model
        self.encoders = nn.ModuleList()
        for enc in encoders:
            self.encoders.append(enc(args.z_size))
        self.decoders = nn.ModuleList()
        for dec in decoders:
            self.decoders.append(dec(args.z_size))
        self.discriminators = nn.ModuleList()
        for dis in discriminators:
            self.discriminators.append(dis(args.z_size, 2))
        self.z_size = args.z_size
        if args.cuda:
            self.cuda()

    def forward(self, inputs):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var = self.encode(inputs)
        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)
        # z0 = z0.to(z_mu)

        reconstructions = self.decode(z0)

        return reconstructions, z_mu, z_var, torch.zeros((z0.shape[0], )).to(z0), z0, None

    def decode(self, z0):
        reconstructions = []
        for dec in self.decoders:
            reconstructions.append(dec(z0))

        return reconstructions

    def discriminate(self, inputs):
        preds = []
        for inp, dis in zip(inputs, self.discriminators):
            if inp is None:
                preds.append(None)
            else:
                pred, _ = dis(inp)
                preds.append(pred)

        return preds

    def discriminate_with_features(self, inputs):
        preds = []
        feats = []
        for inp, dis in zip(inputs, self.discriminators):
            if inp is None:
                preds.append(None)
                feats.append(None)
            else:
                pred, feat = dis(inp)
                preds.append(pred)
                feats.append(feat)

        return preds, feats

    def encode(self, inputs):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        mean_z, var_z = self.infer(inputs)

        return mean_z, var_z

    def infer(self, inputs):
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
            mean_z, var_z = enc(inp)
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

    def freeze_params(self, enc_freeze, dec_freeze, dis_freeze):
        for enc in self.encoders:
            for p in enc.parameters():
                p.requires_grad = enc_freeze

        for dec in self.decoders:
            for p in dec.parameters():
                p.requires_grad = dec_freeze

        for dis in self.discriminators:
            for p in dis.parameters():
                p.requires_grad = dis_freeze


class MCVAEGAN(MVAEGAN):

    def __init__(self, args, encoders, decoders, embeddings):
        super(MCVAEGAN, self).__init__(args, encoders, decoders, [])
        self.discriminator = MultimodalDiscriminator(args.z_size, 2, embeddings)
        self.z_size = args.z_size
        if args.cuda:
            self.cuda()

    def discriminate(self, inputs):
        preds, _ = self.discriminator(inputs)

        return preds

    def discriminate_with_features(self, inputs):
        preds, feats = self.discriminator(inputs)

        return preds, feats

    def freeze_params(self, enc_freeze, dec_freeze, dis_freeze):
        for enc in self.encoders:
            for p in enc.parameters():
                p.requires_grad = enc_freeze

        for dec in self.decoders:
            for p in dec.parameters():
                p.requires_grad = dec_freeze

        for p in self.discriminator.parameters():
            p.requires_grad = dis_freeze


# class Aux(nn.Module):
#     def __init__(self, args, encoders, decoders, discriminators):
#         super(Aux, self).__init__()
#         # CNF model
#         self.decoders = nn.ModuleList()
#         for dec in decoders:
#             self.decoders.append(dec(args.z_size))
#         self.z_size = args.z_size
#         if args.cuda:
#             self.cuda()
#
#     def decode(self, z0):
#         reconstructions = []
#         for dec in self.decoders:
#             reconstructions.append(dec(z0))
#
#         return reconstructions
#
#     def __init__(self):
#         super(Aux,self).__init__()
#
#         self.fc3 = nn.Linear(20,400)
#         self.fc4 = nn.Linear(400,784)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def decode(self,z):
#         z = z.view(-1,20)
#         h3 = self.relu(self.fc3(z))
#         return self.sigmoid(self.fc4(h3))
#     
#     def reparameterize(self, mu, logvar):
#         if self.training:
#           std = logvar.mul(0.5).exp_()
#           eps = Variable(std.data.new(std.size()).normal_())
#           return eps.mul(std).add_(mu)
#         else:
#           return mu
#
#     def dec_params(self):
#         return self.fc3,self.fc4
#
#     def return_weights(self):
#         return self.fc3.weight, self.fc4.weight
#
#     
#     def forward(self,mu,logvar,fc3_weight, fc4_weight):
#         self.fc3.weight = fc3_weight
#         self.fc4.weight = fc4_weight
#         
#         z = self.reparameterize(mu,logvar)
#         #other.fc3,other.fc4 = self.dec_params()
#         #return self.decode(z).view(-1,28,28)
# return self.decode(z)
