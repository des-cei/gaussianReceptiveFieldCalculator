# Pytorch imports
import torch
import torch.nn as nn
import numpy as np

# Leaky neuron parameters
beta = 0.95
threshold = 1.0

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
    def __init__(self):
        super(LeakySurrogate, self).__init__()

        # initialize decay rate beta and threshold
        self.beta = beta
        self.threshold = threshold
        self.spike_gradient = self.ATan.apply


  # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
        spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
        reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
        mem = self.beta * mem + input_ - reset  # Eq (1)
        return spk, mem

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
            ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors  # retrieve the membrane potential
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
            return grad
        

# Izhi neuron parameters
a1 = 1.0
a2 = -0.210
a3 = 0.019
b1 = -1.0/32.0
b2 = 1.0/32.0
b3 = 0.0
c = 0.105
d = 0.412
vth = 0.7

# Leaky neuron model, overriding the backward pass with a custom function
class Izhi(nn.Module):
    def __init__(self):
        super(Izhi, self).__init__()

        # initialize decay rate beta and threshold
        self.threshold = vth
        self.spike_gradient = self.ATan.apply
        

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem, rec):
        spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function

        incr_mem = (a1*torch.mul(mem,mem)+a2*mem-a3*rec + input_)
        incr_rec = (b1*mem-b2*rec+b3)

        reset_mem = (spk * (-mem + c - incr_mem)*spk).detach()   # remove membrane reset from computational graph
        reset_rec = (spk * (d - incr_rec)).detach()              # remove recovery reset from computational graph
        
        mem = mem + 0.1*incr_mem + reset_mem
        rec = rec + 0.1*incr_rec + reset_rec

        # Izhikevich equations
        return spk, mem, rec

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            spk = (mem > 0).float()     # Heaviside on the forward pass: Eq(2)
            ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
            return spk

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors  # retrieve the membrane potential
            grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
            return grad