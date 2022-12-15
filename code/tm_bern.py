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

from logit_normal import *

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def tm_bern_model(D, nTopics, prior_alpha=50, nCells=5, nRegions=3):
    """
    Implementation of Bernoulli topic model.
    
    :param D: Data matrix, [nRegions, nCells] shaped array
    :param nTopics: Number of topics
    :param prior_alpha: Topic cells prior parameter. This is the shape parameter 'alpha' of the Gamma distribution.
    :param nCells: Optional number of cells, used only when D=None to generate a new dataset
    :param nRegions: Optional number of regions, used only when D=None to generate a new dataset
    """
    if D is not None:
        nCells = D.shape[1]
        nRegions = D.shape[0]
    
    # Index matrix on nRegions x nCells
    idx = torch.arange(0,nRegions).unsqueeze(1).repeat(1, nCells)
    
    # Define reusable context managers
    topics_plate = pyro.plate(name='nTopics', size=nTopics, dim=-1)
    phi_regions_plate = pyro.plate(name='phi_nRegions', size=nRegions, dim=-2)
    
    ## Globals
    # Topic-regions prior
    beta = torch.ones([nRegions, nTopics]) * (-1.5)
    # Topic-cells prior
    #alpha = torch.ones(nTopics)/nTopics
    with topics_plate:
        alpha = pyro.sample("alpha", dist.Gamma(prior_alpha, 1.0))
    # Topic-regions distribution
    with topics_plate, phi_regions_plate:
        phi = pyro.sample(name='phi', fn=LogitNormal(beta, 1.5))

    ## Locals
    with pyro.plate(name='nCells', size=nCells):
        # Topic-cells distribution
        theta = pyro.sample(name='theta', fn=dist.Dirichlet(alpha))
        with pyro.plate(name='nRegions', size=nRegions):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            z = pyro.sample(name='z', fn=dist.Categorical(theta), infer={'enumerate' : 'parallel'})
            phi_z = Vindex(phi)[..., idx, z]
            w = pyro.sample(name='w', fn=dist.Bernoulli(phi_z), obs=D)
                 
    obj = dict()
    obj['alpha'] = alpha
    obj['beta'] = beta
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj


def tm_bern_guide(D, nTopics, prior_alpha=50, nCells=5, nRegions=3):
    """
    Guide implementation of Bernoulli topic model.
    Data is a [nRegions, nCells] shaped array.
    
    :param D: Data matrix, [nRegions, nCells] shaped array
    :param nTopics: Number of topics
    :param prior_alpha: Topic cells prior parameter. This is the shape parameter 'alpha' of the Gamma distribution.
    :param nCells: Optional number of cells, used only when D=None to generate a new dataset
    :param nRegions: Optional number of regions, used only when D=None to generate a new dataset
    """
    
    if D is not None:
        nCells = D.shape[1]
        nRegions = D.shape[0]
        
    # Define reusable context managers
    topics_plate = pyro.plate(name='nTopics', size=nTopics, dim=-1)
    phi_regions_plate = pyro.plate(name='phi_nRegions', size=nRegions, dim=-2)
    
    ## Globals
    alpha_a_vi = pyro.param(
        "alpha_a_vi", 
        lambda: dist.LogNormal(loc=3., scale=0.5).sample([nTopics]),
        constraint=constraints.positive)
    alpha_b_vi = pyro.param(
        "alpha_b_vi", 
        lambda: dist.LogNormal(loc=1., scale=0.5).sample([nTopics]),
        constraint=constraints.positive)
    phi_vi = pyro.param(
        name='phi_vi', 
        init_tensor=lambda: dist.Normal(loc=0., scale=0.2).sample([nRegions, nTopics]))
    theta_vi = pyro.param(
        name='theta_vi', 
        init_tensor=lambda: dist.LogNormal(loc=0., scale=0.5).sample([nCells, nTopics]),
        constraint=constraints.positive)
    # Iterate over topics
    with topics_plate:
        alpha = pyro.sample("alpha", dist.Gamma(alpha_a_vi, alpha_b_vi))
    # Iterate over topics and regions
    with topics_plate, phi_regions_plate:
        phi = pyro.sample(name='phi', fn=LogitNormal(phi_vi, 0.2))
    
    # Iterate over cells
    with pyro.plate(name='nCells', size=nCells):
        # Topic-cells distribution
        theta = pyro.sample(name='theta', fn=dist.Dirichlet(theta_vi))
                          
    obj = dict()
    obj['alpha_a_vi'] = alpha_a_vi
    obj['alpha_b_vi'] = alpha_b_vi
    obj['theta_vi'] = theta_vi
    obj['phi_vi'] = phi_vi
    obj['alpha'] = alpha
    obj['theta'] = theta
    obj['phi'] = phi
    obj['D'] = D
    return obj