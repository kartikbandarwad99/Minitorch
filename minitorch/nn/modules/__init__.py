# minitorch/nn/modules/__init__.py
from .linear import Linear
from .activation import Tanh
from .softmax import Softmax
from .embedding import Embedding
from .flatten import Flatten
# from .sequential import Sequential

__all__ = ['Linear', 'Tanh', 'Softmax', 'Embedding', 'Flatten']
