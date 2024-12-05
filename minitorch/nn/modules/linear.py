import minitorch as torch
class Linear:
    def __init__(self,shape):
        self.w = torch.Tensor.randn(shape) * 0.1
        self.b = torch.Tensor.randn((1,shape[-1])) * 0.01
    def __call__(self,other):
        out = other @ self.w + self.b
        return out
  # def __repr__(self):
  #   return f'{self.w} {self.b}'
    def parameters(self):
        """Return a list of parameters in this layer."""
        return [self.w, self.b]

    def zero_grad(self):
        """Reset gradients of parameters to None."""
        self.w.grad = None
        self.b.grad = None
