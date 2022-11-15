"""
Source: https://pyro.ai/examples/lda.html 

This example implements amortized Latent Dirichlet Allocation [1],
demonstrating how to marginalize out discrete assignment variables in a Pyro
model. This model and inference algorithm treat cells as vectors of
categorical variables (vectors of region ids), and collapses region-topic
assignments using Pyro's enumeration. We use PyTorch's reparametrized Gamma and
Dirichlet distributions [2], avoiding the need for Laplace approximations as in
[1]. Following [1] we use the Adam optimizer and clip gradients.

**References:**

[1] Akash Srivastava, Charles Sutton. ICLR 2017.
    "Autoencoding Variational Inference for Topic Models"
    https://arxiv.org/pdf/1703.01488.pdf
[2] Martin Jankowiak, Fritz Obermeyer. ICML 2018.
    "Pathwise gradients beyond the reparametrization trick"
    https://arxiv.org/pdf/1806.01851.pdf
"""


import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def amortized_lda_model(D=None, nTopics=5, nRegions=100, batch_size=None, nCells=50, N=10):
    """
    This is a fully generative model of a batch of cells.
    Data is a [nCounts_per_cell, nCells] shaped array of region ids
    (specifically it is not a histogram). We assume in this simple example
    that all cells have the same number of nCounts.
    
    :param D: Data matrix
    :param nTopics: Number of topics
    :param nRegions: Number of regions (vocabulary size)
    :param batch_size: Batch size when performing inference
    :param nCells: Optional number of cells, used only when D=None to generate a new dataset
    :param N: Optional nCounts per cell, used only when D=None to generate a new dataset
    """
    
    if D is not None:
        nCells = D.shape[1]
        N = D.shape[0]
        
    # Globals
    b = torch.ones(nRegions)/nRegions
    with pyro.plate("nTopics", nTopics):
        a = pyro.sample("a", dist.Gamma(1.0 / nTopics, 1.0))
        phi = pyro.sample("phi", dist.Dirichlet(b))
        
    # Locals
    with pyro.plate("nCells", nCells) as ind:
        if D is not None:
            assert D.shape == (N, nCells)
            D = D[:, ind]
        theta = pyro.sample("theta", dist.Dirichlet(a))
        with pyro.plate("nCounts", N):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            z = pyro.sample("z", dist.Categorical(theta), infer={"enumerate": "parallel"})
            D = pyro.sample("w", dist.Categorical(phi[z]), obs=D)
                          
    obj = dict()
    obj['alpha'] = a
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj


def nn_predictor(nTopics, nRegions, layer_sizes):
    """
    We will use amortized inference of the local topic variables, achieved 
    by a multi-layer perceptron. We'll wrap the guide in an nn.Module.
    """
    layer_sizes = (
        [nRegions]
        + [int(s) for s in layer_sizes.split("-")]
        + [nTopics]
    )
    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layer = nn.Linear(in_size, out_size)
        layer.weight.data.normal_(0, 0.001)
        layer.bias.data.normal_(0, 0.001)
        layers.append(layer)
        layers.append(nn.Sigmoid())
    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


def amortized_lda_guide(pred, D, nTopics, nRegions, batch_size=None, nCells=50, N=10):
    """
    This is a fully generative guide of a batch of cells.
    Data is a [nCounts_per_cell, nCells] shaped array of region ids
    (specifically it is not a histogram). We assume in this simple example
    that all cells have the same number of nCounts.
    
    :param pred: NN predictor object
    :param D: Data matrix
    :param nTopics: Number of topics
    :param nRegions: Number of regions (vocabulary size)
    :param batch_size: Batch size when performing inference
    :param nCells: Optional number of cells, used only when D=None to generate a new dataset
    :param N: Optional nCounts per cell, used only when D=None to generate a new dataset
    """
    if D is not None:
        nCells = D.shape[1]
        N = D.shape[0]
        
    # Use conjugate guide for global variables
    a_vi = pyro.param("a_vi", lambda: torch.ones(nTopics), 
                      constraint = constraints.positive)
    b_vi = pyro.param("b_vi", lambda: torch.ones(nTopics, nRegions), 
                      constraint = constraints.greater_than(0.5))
    
    with pyro.plate("nTopics", nTopics):
        a = pyro.sample("a", dist.Gamma(a_vi, 1.0))
        phi = pyro.sample("phi", dist.Dirichlet(b_vi))
    
    # Use an amortized guide for local variables.
    pyro.module("pred", pred)
    with pyro.plate("nCells", nCells, batch_size) as ind:
        D = D[:, ind]
        # The neural network will operate on histograms rather than region
        # index vectors, so we'll convert the raw data to a histogram.
        freq = torch.zeros(nRegions, ind.size(0)).scatter_add(0, D, torch.ones(D.shape))
        nn_theta = pred(freq.transpose(0, 1))
        theta = pyro.sample("theta", dist.Delta(nn_theta, event_dim=1))
                          
    obj = dict()
    obj['alpha'] = a
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj


def fit_amortized_lda(D, nTopics, nRegions, nSteps = 1000, batch_size=64, 
                      layer_sizes = "100-100", lr = 0.01, seed = 1):
    """
    Fit Amortized LDA
    """
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    # We'll fit using SVI.
    logging.info("-" * 40)
    logging.info("Fitting {} cells".format(D.shape[1]))
    # Create NN predictor
    pred = nn_predictor(nTopics, nRegions, layer_sizes)
    # Our guide
    guide = functools.partial(amortized_lda_guide, pred)

    elbo = pyro.infer.TraceEnum_ELBO
    elbo = elbo(max_plate_nesting=2)
    optim = pyro.optim.ClippedAdam({"lr": lr})
    svi = pyro.infer.SVI(amortized_lda_model, guide, optim, elbo)
    losses = []
    
    logging.info("Step\tLoss")
    for step in range(nSteps):
        loss = svi.step(D=D, nTopics=nTopics, nRegions=nRegions, batch_size=batch_size)
        if step % 100 == 0:
            logging.info("{: >5d}\t{}".format(step, loss))
        losses.append(loss)
    logging.info("final loss = {}".format(loss))

    obj = dict()
    obj['losses'] = losses
    obj['guide'] = guide
    return obj
