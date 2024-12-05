from minitorch import Tensor
class Embedding:
    def __init__(self,shape):
        self.emb = Tensor.randn(shape=shape)
    def __call__(self,x):
        return self.emb[x]
    def parameters(self):
        """Return a list of parameters in this layer."""
        return [self.emb]
    def zero_grad(self):
        """Reset gradients of parameters to None."""
        self.emb.grad = None

