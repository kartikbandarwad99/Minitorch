class Flatten:
    def __init__(self):
        pass
        
    def __call__(self, x):
        return x.view(x.shape[0], -1)