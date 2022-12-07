"""

Pyro implementation of mixed-membership model by marginalizing out discrete assignment 
variables and with Bernoulli observation likelihood. 
This model and inference algorithm treat cells as vectors of
binary variables (vectors of regions), and collapses region-topic
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

# Logit-Normal distribution, currently Pyro does not implement this.
def LogitNormal(mu, s): 
    base_dist = pyro.distributions.Normal(loc = mu, scale = s)
    response_dist = pyro.distributions.TransformedDistribution(
        base_distribution=base_dist, transforms=torch.distributions.transforms.SigmoidTransform())
    return response_dist


def tm_bern_model(D=None, nTopics=2, nCells=5, nRegions = 3):
    """
    This is a fully generative model of a batch of cells.
    
    :param D: Data matrix, [nRegions, nCells] shaped array
    :param nTopics: Number of topics
    :param nCells: Optional number of cells, used only when D=None to generate a new dataset
    :param nRegions: Optional number of regions, used only when D=None to generate a new dataset
    """
    if D is not None:
        nCells = D.shape[1]
        nRegions = D.shape[0]
    
    # Index matrix on nRegions x nTopics
    idx = torch.arange(0,nRegions).unsqueeze(1).repeat(1, nCells)
    
    # Define reusable context managers
    topics_plate = pyro.plate(name="nTopics", size=nTopics, dim=-1)
    phi_regions_plate = pyro.plate(name="phi_nRegions", size=nRegions, dim=-2)
    
    ## Globals
    # Topic-regions prior
    b = torch.ones([nRegions, nTopics])/10.
    # Topic-cells prior
    alpha = torch.ones(nTopics)/nTopics
    #with topics_plate:
    #    alpha = pyro.sample("alpha", dist.Gamma(1.0 / nTopics, 1.0))
    
    # Topic-regions distribution
    with phi_regions_plate, topics_plate:
        phi = pyro.sample(name="phi", fn=LogitNormal(b, 1))
    ## Locals
    with pyro.plate(name="nCells", size=nCells):
        # Topic-cells distribution
        theta = pyro.sample(name="theta", fn=dist.Dirichlet(alpha))
        with pyro.plate(name="nRegions", size=nRegions):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            z = pyro.sample(name="z", fn=dist.Categorical(theta), infer={"enumerate" : "parallel"})
            print(z.shape)
            print(z)
            tmp = phi[idx, z]
            print(tmp.shape)
            print(tmp)
            print(phi.shape)
            print(phi)
            w = pyro.sample(name="w", fn=dist.Bernoulli(phi[idx, z]), obs=D)
                 
    obj = dict()
    obj['alpha'] = alpha
    obj['beta'] = b
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj


def tm_bern_guide(D, nTopics=2, nCells=5, nRegions=3):
    """
    Guide implementation of Bernoulli topic model.
    Data is a [nRegions, nCells] shaped array.
    
    :param D: Data matrix, [nRegions, nCells] shaped array
    :param nTopics: Number of topics
    :param nCells: Optional number of cells, used only when D=None to generate a new dataset
    :param nRegions: Optional number of regions, used only when D=None to generate a new dataset
    """
    
    if D is not None:
        nCells = D.shape[1]
        nRegions = D.shape[0]
        
    # Define reusable context managers
    topics_plate = pyro.plate(name="nTopics", size=nTopics, dim=-1)
    phi_regions_plate = pyro.plate(name="phi_nRegions", size=nRegions, dim=-2)
    
    ## Globals
    phi_vi = pyro.param(
        "phi_vi", 
        lambda: dist.LogNormal(loc=0., scale=1).sample([nRegions, nTopics]),
        constraint=constraints.positive)
    theta_vi = pyro.param(
        "theta_vi", 
        lambda: dist.LogNormal(loc=0., scale=0.5).sample([nCells, nTopics]),
        constraint=constraints.positive)
    
    # Iterate over topics and regions
    with phi_regions_plate, topics_plate:
        phi = pyro.sample("phi", dist.Beta(phi_vi, 2))
    
    # Iterate over cells
    with pyro.plate("nCells", nCells):
        # Topic-cells distribution
        theta = pyro.sample("theta", dist.Dirichlet(theta_vi))
                          
    obj = dict()
    obj['theta_vi'] = theta_vi
    obj['phi_vi'] = phi_vi
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj
