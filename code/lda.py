"""

Pyro implementation of LDA model by marginalizing out discrete assignment 
variables. This model and inference algorithm treat cells as vectors of
categorical variables (vectors of region ids), and collapses region-topic
assignments using Pyro's enumeration.
Following [1] we use the Adam optimizer and clip gradients.

**References:**

[1] Akash Srivastava, Charles Sutton. ICLR 2017.
    "Autoencoding Variational Inference for Topic Models"
    https://arxiv.org/pdf/1703.01488.pdf
"""


import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.ops.indexing import Vindex

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

def lda_model(D=None, nTopics=5, nRegions=100, nCells=50, N=10):
    """
    This is a fully generative model of a batch of cells.
    Data is a [nCounts_per_cell, nCells] shaped array of region ids. 
    We assume in this simple example that all cells have the same number of nCounts.
    
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
    
    ## Globals
    # Topic-cells prior
    a = pyro.param("a", torch.ones(nTopics))
    # Topic-regions prior
    b = pyro.param("b", torch.ones(nRegions) / 100)
    # For each topic generate topic-region distribution
    with pyro.plate("nTopics", nTopics):
        phi = pyro.sample("phi", dist.Dirichlet(b))
    
    ## Locals
    with pyro.plate("nCells", nCells):
        # Topic-cells distribution
        theta = pyro.sample("theta", dist.Dirichlet(a))
        with pyro.plate("nCounts", N):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            z = pyro.sample("z", dist.Categorical(theta), infer={"enumerate" : "parallel"})
            phi_z = Vindex(phi)[z, :]
            w = pyro.sample("w", dist.Categorical(phi_z), obs=D)
            
    obj = dict()
    obj['alpha'] = a
    obj['beta'] = b
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj


def lda_guide(D, nTopics, nRegions, nCells=50, N=10):
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
    
    ## Globals
    a_vi = pyro.param("a_vi", lambda: torch.ones(nTopics) + torch.Tensor(list(range(nTopics))), 
                      constraint=constraints.positive)
    b_vi = pyro.param("b_vi", lambda: torch.ones(nRegions), 
                      constraint=constraints.positive)
    
    # Iterate over topics
    with pyro.plate("nTopics", nTopics):
        phi = pyro.sample("phi", dist.Dirichlet(b_vi))
    
    # Iterate over cells
    with pyro.plate("nCells", nCells):
        # Topic-cells distribution
        theta = pyro.sample("theta", dist.Dirichlet(a_vi))
                          
    obj = dict()
    obj['a_vi'] = a_vi
    obj['b_vi'] = b_vi
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj


def fit_lda(D, nTopics, nRegions, nSteps = 1000, lr = 0.01, seed = 1):
    """
    Fit Amortized LDA
    """
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    # We'll fit using SVI.
    logging.info("-" * 40)
    logging.info("Fitting {} cells".format(D.shape[1]))

    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=2)
    optim = pyro.optim.ClippedAdam({"lr": lr})
    svi = pyro.infer.SVI(lda_model, lda_guide, optim, elbo)
    losses = []
    
    logging.info("Step\tLoss")
    for step in range(nSteps):
        loss = svi.step(D=D, nTopics=nTopics, nRegions=nRegions)
        if step % 100 == 0:
            logging.info("{: >5d}\t{}".format(step, loss))
        losses.append(loss)
    logging.info("final loss = {}".format(loss))

    obj = dict()
    obj['losses'] = losses
    return obj