# minitorch/__init__.py
from contextlib import contextmanager

_grad_enabled = True

@contextmanager
def no_grad():
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev

from .Tensor import Tensor
from .nn import *