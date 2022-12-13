## Adapted from Pytorch code for LogNormal distribution:
# https://github.com/pytorch/pytorch/blob/master/torch/distributions/log_normal.py

import pyro
import torch
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from pyro.distributions.torch_distribution import TorchDistributionMixin

# Logit-Normal distribution, currently Pyro does not implement this.
#def LogitNormal_old(mu, s): 
#    base_dist = pyro.distributions.Normal(loc = mu, scale = s)
#    response_dist = pyro.distributions.TransformedDistribution(
#        base_distribution=base_dist, transforms=torch.distributions.transforms.SigmoidTransform())
#    return response_dist

#__all__ = ['LogitNormal']

class LogitNormalTorch(TransformedDistribution):
    r"""
    Creates a logit-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::
        X ~ Normal(loc, scale)
        Y = logit(X) ~ LogitNormal(loc, scale)
    Example::
        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = LogitNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # logit-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])
    Args:
        loc (float or Tensor): mean of logit of distribution
        scale (float or Tensor): standard deviation of logit of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = pyro.distributions.Normal(loc, scale, validate_args=validate_args)
        super(LogitNormalTorch, self).__init__(
            base_dist, 
            torch.distributions.transforms.SigmoidTransform(), 
            validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogitNormalTorch, _instance)
        return super(LogitNormalTorch, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale
    

    
class LogitNormal(LogitNormalTorch, TorchDistributionMixin):
    pass
