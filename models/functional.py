'''
Script provides functional interface for Mish activation function.
'''

# import pytorch
import torch
import torch.nn.functional as F


# @torch.jit.script
def beta_mish(input, beta=1.5):
    """
    Applies the β mish function element-wise:
        .. math::
            \\beta mish(x) = x * tanh(ln((1 + e^{x})^{\\beta}))
    See additional documentation for :mod:`echoAI.Activation.Torch.beta_mish`.
    """
    return input * torch.tanh(torch.log(torch.pow((1 + torch.exp(input)), beta)))


def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

