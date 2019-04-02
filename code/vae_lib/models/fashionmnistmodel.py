#this is from FFJORD
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from code.vae_lib.models.train_misc import build_model_tabular
from .VAE import VAE


def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.z_size,)


def concat_layer_num_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim + out_dim
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class CNFVAE(VAE):

    def __init__(self, args):
        super(CNFVAE, self).__init__(args)
        print ("CNF created")
        self.experts = ProductOfExperts()
        # CNF model
        self.cnf = build_model_tabular(args, args.z_size)
        self.text_encoder = TextEncoder(args.z_size)
        self.text_decoder = TextDecoder(args.z_size)
        self.image_encoder = ImageEncoder(args.z_size)
        self.image_decoder = ImageDecoder(args.z_size)
        self.z_size = args.z_size
        if args.cuda:
            self.cuda()


    def forward(self, image=None, text=None):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        z_mu, z_var = self.encode(image, text)
        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)
        z0 = z0.to(torch.float32).cuda()
        if image is not None:
            zero = torch.zeros(image.shape[0], 1).to(torch.float32).cuda()
        elif text is not None:
            zero = torch.zeros(text.shape[0], 1).to(torch.float32).cuda()

        zk, delta_logp = self.cnf(z0, zero)  # run model forward
        image_rec = self.image_decoder(zk)
        text_rec = self.text_decoder(zk)

        return image_rec, text_rec, z_mu, z_var, -delta_logp.view(-1), z0, zk

    def encode(self, image=None, text=None):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        mean_z, var_z = self.infer(image, text)

        return mean_z, var_z


    def infer(self, image=None, text=None):

        batch_size = image.size(0) if image is not None else text.size(0)

        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.z_size),
                                  use_cuda=use_cuda)

        if image is not None:
            mean_z, var_z = self.image_encoder(image)
            mu = torch.cat((mu, mean_z.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, var_z.unsqueeze(0)), dim=0)

        if text is not None:
            mean_z, var_z = self.text_encoder(text)
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


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)

        return pd_mu, pd_logvar


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


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.n_latents = n_latents
        self.upsampler = nn.Sequential(
            nn.Linear(n_latents, 512),
            Swish(),
            nn.Linear(512, 128 * 7 * 7),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |n_latents|
        z = self.upsampler(z)
        z = z.view(-1, 128, 7, 7)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Embedding(10, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x)
        return x[:, :n_latents], x[:, n_latents:]


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latents, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, 10))

    def forward(self, z):
        z = self.net(z)
        return z  # NOTE: no softmax here. See train.py