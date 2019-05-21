#MVAE from the multimodal VAE paper
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from ml_utils.LSTM import LSTMLayer
# from ml_utils.LSTM import LSTMLayer


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder  = TextEncoder(n_latents)
        self.text_decoder  = TextDecoder(n_latents)
        self.experts       = ProductOfExperts()
        self.n_latents     = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, image=None, text=None):
        mu, logvar = self.infer(image, text)
        # reparametrization trick to sample
        z          = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        img_recon  = self.image_decoder(z)
        txt_recon  = self.text_decoder(z)
        return img_recon, txt_recon, mu, logvar

    def infer(self, image=None, text=None):
        batch_size = image.size(0) if image is not None else text.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents),
                                  use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_encoder(image)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if text is not None:
            txt_mu, txt_logvar = self.text_encoder(text)
            mu     = torch.cat((mu, txt_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, txt_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)

        return mu, logvar


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class ImageDiscriminator(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, n_classes):
        super(ImageDiscriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_latent = nn.Linear(512, n_latents)
        self.out = nn.Linear(n_latents, n_classes)
        self.swish = Swish()
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        h_latent = self.swish(self.fc_latent(h))
        out = self.logsoftmax(self.out(self.dropout(h_latent)))
        return out, h_latent


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)  # NOTE: no sigmoid here. See train.py


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Embedding(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class TextReprEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextReprEncoder, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class TextDiscriminator(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, n_classes):
        super(TextDiscriminator, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_latent = nn.Linear(512, n_latents)
        self.out = nn.Linear(n_latents, n_classes)
        self.swish = Swish()
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        emb = self.fc1(x)
        h = self.swish(emb)
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc_latent(h))
        return self.logsoftmax(self.out(self.dropout(h))), h


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents):
        super(TextDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return F.softmax(self.fc4(h)) + 0.001  # NOTE: no softmax here. See train.py


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


class MultimodalDiscriminator(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, n_latents, n_classes, embedders):
        super(MultimodalDiscriminator, self).__init__()
        self.embedders = nn.ModuleList()
        for emb in embedders:
            self.embedders.append(emb)
        self.fc1 = nn.Linear(len(embedders) * 512, 512)
        self.fc_latent = nn.Linear(512, n_latents)
        self.out = nn.Linear(n_latents, n_classes)
        self.swish = Swish()
        self.logsoftmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        batch_size = 1
        device = 'cpu'
        for inp in inputs:
            if inp is not None:
                batch_size = inp.shape[0]
                device = inp.device
                break
        embeddings = []
        for inp, emb in zip(inputs, self.embedders):
            if inp is None:
                embeddings.append(torch.zeros(batch_size, 512).to(device))
            else:
                embeddings.append(emb(inp))

        embeddings = torch.cat(embeddings, dim=1)

        h = self.swish(self.fc1(embeddings))
        h2 = self.swish(self.fc_latent(h))
        out = self.logsoftmax(self.out(self.dropout(h2)))
        return out, h2


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# class LSTMEncoder(nn.Module):
#     """Parametrizes p(x|z).
#
#     @param n_latents: integer
#                       number of latent dimensions
#     """
#     def __init__(self, input_size, hidden_size, n_latents):
#         super().__init__()
#         lstm = LSTMLayer(input_size, hidden_size, 1)
#         self.out1 = nn.Linear(hidden_size, n_latents)
#         self.out2 = nn.Linear(hidden_size, n_latents)
#         self.swish = Swish()
#
#     def forward(self, z):
#         h = self.swish(self.fc1(z))
#         h = self.swish(self.fc2(h))
#         h = self.swish(self.fc3(h))
#         return self.fc4(h)  # NOTE: no sigmoid here. See train.py
#
class SeqMLP(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, input_size, output_sizes, layers_sizes=[512, 512], dropout_prob=0):
        super().__init__()

        if not isinstance(output_sizes, list):
            output_sizes = [output_sizes]

        self.output_sizes = output_sizes
        self.layers_sizes = layers_sizes
        self.dropout_prob = dropout_prob
        self.layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.layers.append(nn.Linear(input_size, self.layers_sizes[0]))

        count = 1
        while count <= len(self.layers_sizes) - 1:
            print(count)
            print(self.layers_sizes[count - 1], self.layers_sizes[count])
            self.layers.append(nn.Linear(self.layers_sizes[count - 1], self.layers_sizes[count]))
            count += 1
        for out_size in output_sizes:
            self.out_layers.append(nn.Linear(self.layers_sizes[-1], out_size))
        self.swish = Swish()

    def forward(self, input):
        if type(input) == list:
            input = torch.cat(input, dim=1)
        results = input
        for i in range(len(self.layers)):
            if i > 0:
                results = self.dropout(results)
            results = self.swish(self.layers[i](results))
        outputs = []
        for out_layer in self.out_layers:
            outputs.append(out_layer(results))

        if len(outputs) > 1:
            return tuple(outputs)
        else:
            return outputs[0]


class RecurrentState(LSTMLayer):
    def __init__(self, input_size, hidden_size,
                 verbose=1, random_seed=10, dropout_prob=0.0, device=None):
        super().__init__(input_size, hidden_size, 1, verbose=verbose, random_seed=random_seed, dropout_prob=dropout_prob, device=device)

    def forward(self, input):
        if type(input) == list:
            input = torch.cat(input, dim=1)

        if self.verbose > 1:
            print('Hidden shape', self.hidden.shape)
            print('Input shape', input.shape)
        combined = torch.cat((input, self.hidden), 1)
        forget = F.sigmoid(self.f_gate(combined))
        input = F.sigmoid(self.i_gate(combined))
        cell_new = F.tanh(self.c_gate(combined))
        self.hidden = F.sigmoid(self.o_gate(combined))
        self.cell = (forget * self.cell) + (input * cell_new)
        self.hidden = self.hidden * F.tanh(self.cell)
        self.hidden = self.dropout(self.hidden)

        return self.hidden


class StandardNormalization(nn.Module):
    def __init__(self, mean, std):
        super(StandardNormalization, self).__init__()
        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

    def forward(self, input):
        return (input - self.mean) / self.std

    def scale_back(self, input):
        return (input * self.std) + self.mean
