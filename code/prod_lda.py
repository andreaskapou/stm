"""
Source: https://pyro.ai/examples/prodlda.html

This example implements product Latent Dirichlet Allocation [1].

**References:**

[1] Akash Srivastava, Charles Sutton. ICLR 2017.
    "Autoencoding Variational Inference for Topic Models"
    https://arxiv.org/pdf/1703.01488.pdf
"""

import functools
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

class Encoder(nn.Module):
    """
    Base class for the encoder net, used in the guide
    """
    def __init__(self, nRegions, nTopics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(nRegions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, nTopics)
        self.fclv = nn.Linear(hidden, nTopics)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(nTopics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(nTopics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    """
    Base class for the decoder net, used in the model
    """
    def __init__(self, nRegions, nTopics, dropout):
        super().__init__()
        self.phi = nn.Linear(nTopics, nRegions, bias=False)
        self.bn = nn.BatchNorm1d(nRegions, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is sigmoid(phi * theta)
        return F.softmax(self.bn(self.phi(inputs)), dim=1)
    

class ProdLDA(nn.Module):
    """
    Base class for ProdLDA Pyro implementation.
    """
    def __init__(self, nRegions, nTopics, hidden, dropout):
        super().__init__()
        self.nRegions = nRegions
        self.nTopics = nTopics
        self.encoder = Encoder(nRegions, nTopics, hidden, dropout)
        self.decoder = Decoder(nRegions, nTopics, dropout)

    def model(self, D):
        """
        Pyro model corresponds to the decoder network
        """
        pyro.module("decoder", self.decoder)
        with pyro.plate("nCells", D.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_loc = D.new_zeros((D.shape[0], self.nTopics))
            logtheta_scale = D.new_ones((D.shape[0], self.nTopics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # conditional distribution of 𝑤𝑛 is defined as
            # w|phi,theta ~ Categorical(𝜎(phi * theta))
            count_param = self.decoder(theta)
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of nCounts across cells can vary,
            # we will use the maximum count accross cells here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count = int(D.sum(-1).max())
            pyro.sample('w', dist.Multinomial(total_count, count_param), obs=D)
            
        return theta

    def guide(self, D):
        """
        Pyro guide corresponds to the encoder network
        """
        pyro.module("encoder", self.encoder)
        with pyro.plate("nCells", D.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_loc, logtheta_scale = self.encoder(D)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            
        theta = F.softmax(logtheta, -1)
        return theta

    def get_phi(self):
        """ 
        Return the phi matrix, whose elements are the weights of the FC layer on the decoder
        """
        return self.decoder.phi.weight.cpu().detach().T


def fit_prod_lda(D, nTopics, nEpochs = 20, batch_size=64, lr = 0.01, seed = 123):
    """
    Fit prodLDA
    """
    
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    # We'll fit using SVI.
    logging.info("-" * 40)
    logging.info("Fitting {} cells".format(D.shape[0]))
    
    prodLDA = ProdLDA(
        nRegions = D.shape[1],
        nTopics=nTopics,
        hidden=100,
        dropout=0.2
    )
    elbo = pyro.infer.TraceMeanField_ELBO()
    optim = pyro.optim.ClippedAdam({"lr": lr})
    svi = pyro.infer.SVI(model=prodLDA.model, guide=prodLDA.guide, optim=optim, loss=elbo)
    num_batches = int(math.ceil(D.shape[0] / batch_size))
    losses = []
    
    logging.info("Step\tLoss")
    for epoch in range(nEpochs):
        running_loss = 0.0
        for i in range(num_batches):
            batch = D[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(D=batch)
            running_loss += loss / batch.size(0)
        logging.info("{: >5d}\t{}".format(epoch, running_loss))
        losses.append(running_loss)
    logging.info("final loss = {}".format(running_loss))

    obj = dict()
    obj['losses'] = losses
    obj['prodLDA'] = prodLDA
    return obj