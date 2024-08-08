"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            in_features, out_features, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(
            out_features, 1, requires_grad=True, device=device, dtype=dtype).transpose()) if bias else None
        # END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        return out + (self.bias.broadcast_to(out.shape) if self.bias is not None else 0)
        # END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        # END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.relu(x)
        # END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module.forward(x)
        return x
        # END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION
        one_hot_y = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        return ops.summation(ops.logsumexp(logits, (1,)) / logits.shape[0]) - ops.summation(one_hot_y * logits / logits.shape[0])
        # END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(
            dim, requires_grad=True, device=device, dtype=dtype))
        self.running_mean = init.zeros(
            dim, requires_grad=False, device=device, dtype=dtype)
        self.running_var = init.ones(
            dim, requires_grad=False, device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        assert x.shape[1] == self.dim
        batch_size, num_features = x.shape
        if self.training:
            # use batch average
            # NOTE: here we don't just set x = x.data to detach x because we do need
            #       x's computational graph to backpropate through the self.weight and bias below.
            mean_vec_1d = ops.divide_scalar(ops.summation(x, axes = 0), batch_size)
            mean_row_vec = ops.reshape(mean_vec_1d, (1, num_features))
            mean = ops.broadcast_to(mean_row_vec, x.shape)
            var_vec_1d = ops.divide_scalar(ops.summation(ops.power_scalar(x - mean, 2), axes = 0), batch_size)
            var_row_vec = ops.reshape(var_vec_1d, (1, num_features))
            var = ops.broadcast_to(var_row_vec, x.shape)
            
            # here we calculate the running average of means and variances.
            # NOTE: here we need to detach the mean and variance tensor calculated from our data before
            #       calculating the running mean and running variance, or else it will build up giant
            #       computational graphs and we will run out of memory.
            # self.running_mean = ops.mul_scalar(self.running_mean, 1 - self.momentum) + ops.mul_scalar(mean_vec_1d, self.momentum)
            # self.running_var = ops.mul_scalar(self.running_var, 1 - self.momentum) + ops.mul_scalar(var_vec_1d, self.momentum)
            self.running_mean = ops.mul_scalar(self.running_mean, 1 - self.momentum) + ops.mul_scalar(mean_vec_1d.data, self.momentum)
            self.running_var = ops.mul_scalar(self.running_var, 1 - self.momentum) + ops.mul_scalar(var_vec_1d.data, self.momentum)
 
            std = ops.power_scalar(var + self.eps, 1/2)

            # this will backpropagate through self.weight and self.bias.
            return ops.broadcast_to(ops.reshape(self.weight, (1, num_features)), x.shape) * ops.divide(x - mean, std) + ops.broadcast_to(ops.reshape(self.bias, (1, num_features)), x.shape)
        else:
            # use running average
            mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape)
            var = ops.broadcast_to(ops.reshape(self.running_var, (1, self.dim)), x.shape)
            std = ops.power_scalar(var + self.eps, 1/2)

            # NOTE: backpropagate through the the weights and biases if we are training, otherwise detach weights and biases.
            return ops.broadcast_to(ops.reshape(self.weight.data, (1, num_features)), x.shape) * ops.divide(x - mean, std) + ops.broadcast_to(ops.reshape(self.bias.data, (1, num_features)), x.shape)
        # END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose(
            (2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device, dtype=dtype))
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        mean = (x.sum((1,)) /
                x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x - mean)**2).sum((1,)) /
               x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        deno = (var + self.eps)**0.5
        return self.weight.broadcast_to(x.shape) * (x - mean)/deno + self.bias.broadcast_to(x.shape)
        # END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        if self.training == False:
            return x
        mask = init.randb(*x.shape, p = 1 - self.p, dtype="float32", device=x.device) # 1 - p entries to be 1, p entries to be 0
        return ops.divide_scalar(ops.multiply(x, mask), 1 - self.p)
        # END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return self.fn.forward(x) + x
        # END YOUR SOLUTION
