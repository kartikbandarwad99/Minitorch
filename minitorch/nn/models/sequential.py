class Sequential:
    def __init__(self,layers):
        self.layers = layers
    def __call__(self,inp):
        x = inp
        for i in self.layers:
            x = i(x)
        return x

    def parameters(self):
        """Aggregate parameters from all layers that have a `parameters` method."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def zero_grad(self):
        """Reset gradients for all parameters in the model."""
        for param in self.parameters():
            param.grad = None

    def predict(self):
        # with 
        pass