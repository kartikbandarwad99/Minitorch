import numpy as np
from .. import _grad_enabled  
class Tensor:
    def __init__(self, data, children=None, op='', requires_grad=True):
        self.data = np.array(data, dtype='float64') if not isinstance(data, np.ndarray) else data
        self.grad = None
        self.op = op
        self._backward = lambda: None
        self.children = children if children is not None else []
        self.shape = self.data.shape
        self.requires_grad = requires_grad and _grad_enabled


    def __repr__(self):
        return f'Tensor:{self.data}'

    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, children=[self, other], op='+')

            def _backward(axis=0, keepdims=True):
                # print('add backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                self.grad += out.grad

                grad_other = out.grad
                # Identify axes where other has size 1 and out.grad has size >1
                axes_other = tuple(
                    i for i in range(grad_other.ndim)
                    if other.data.shape[i] == 1 and grad_other.shape[i] > 1
                )
                if axes_other:
                    grad_other = grad_other.sum(axis=axes_other, keepdims=True)
                other.grad += grad_other

            out._backward = _backward
            return out
        else:
            # Scalar addition 
            out = Tensor(self.data + other, children=[self], op='+')

            def _backward():
                # print('scalar add backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad

            out._backward = _backward
            return out

    def __sub__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, children=[self, other], op='-')

            def _backward():
                # print('sub backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                self.grad += out.grad

                grad_other = -out.grad
                # Handle broadcasting for other.grad
                axes_other = tuple(
                    i for i in range(grad_other.ndim)
                    if other.data.shape[i] == 1 and grad_other.shape[i] > 1
                )
                if axes_other:
                    grad_other = grad_other.sum(axis=axes_other, keepdims=True)
                other.grad += grad_other

            out._backward = _backward
            return out
        else:
            # Scalar subtraction
            out = Tensor(self.data - other, children=[self], op='-')

            def _backward():
                # print('scalar sub backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad

            out._backward = _backward
            return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, children=[self, other], op='*')

            def _backward():
                # print('mul backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                grad_self = other.data * out.grad
                grad_other = self.data * out.grad

                # Handle broadcasting for self.grad
                axes_self = tuple(
                    i for i in range(grad_self.ndim)
                    if self.data.shape[i] == 1 and grad_self.shape[i] > 1
                )
                if axes_self:
                    grad_self = grad_self.sum(axis=axes_self, keepdims=True)
                self.grad += grad_self

                # Handle broadcasting for other.grad
                axes_other = tuple(
                    i for i in range(grad_other.ndim)
                    if other.data.shape[i] == 1 and grad_other.shape[i] > 1
                )
                if axes_other:
                    grad_other = grad_other.sum(axis=axes_other, keepdims=True)
                other.grad += grad_other

            out._backward = _backward
            return out
        else:
            # Scalar multiplication
            out = Tensor(self.data * other, children=[self], op='*')

            def _backward():
                # print('scalar mul backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += other * out.grad

            out._backward = _backward
            return out

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data / other.data, children=[self, other], op='/')

            def _backward():
                # print('division backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                grad_self = (1 / other.data) * out.grad
                grad_other = (-self.data / (other.data ** 2)) * out.grad

                # Handle broadcasting for self.grad
                axes_self = tuple(
                    i for i in range(grad_self.ndim)
                    if self.data.shape[i] == 1 and grad_self.shape[i] > 1
                )
                if axes_self:
                    grad_self = grad_self.sum(axis=axes_self, keepdims=True)
                self.grad += grad_self

                # Handle broadcasting for other.grad
                axes_other = tuple(
                    i for i in range(grad_other.ndim)
                    if other.data.shape[i] == 1 and grad_other.shape[i] > 1
                )
                if axes_other:
                    grad_other = grad_other.sum(axis=axes_other, keepdims=True)
                other.grad += grad_other

            out._backward = _backward
            return out
        else:
            # Scalar division
            out = Tensor(self.data / other, children=[self], op='/')

            def _backward():
                # print('scalar division backward called')
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 / other) * out.grad

            out._backward = _backward
            return out

    def __neg__(self):
        out = Tensor(-self.data, children=[self], op='neg')

        def _backward():
            # print('neg backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, children=[self, other], op='@')

        def _backward():
            # print('matmul backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, children=[self], op='tanh')

        def _backward():
            # print('tanh backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += (1 - out_data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, children=[self], op='exp')

        def _backward():
            # print('exp backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += out.data * out.grad 

        out._backward = _backward
        return out

    def log(self):
        out_data = np.log(self.data + 1e-12)  # Adding epsilon to avoid log(0)
        out = Tensor(out_data, children=[self], op='log')

        def _backward():
            # print('log backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += (1 / (self.data + 1e-12)) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, children=[self], op='sum')
        def _backward():
            # print('sum backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = out.grad  # This could be scalar or array
            # print(f'grad.shape: {grad.shape}')
            # print(f'self.grad.shape: {self.grad.shape}')
            # print(f'self.data.shape: {self.data.shape}')
            if axis is None:
                # print('Axis is None')
                if isinstance(grad, np.ndarray):
                    grad = grad.item()
                grad = np.ones_like(self.data) * grad
            else:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                
                if keepdims:
                    grad = np.broadcast_to(grad, self.data.shape)
                else:
                    for a in sorted(axes):
                        grad = np.expand_dims(grad, a)
                    grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, children=[self], op='max')

        def _backward():
            # print('max backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = np.zeros_like(self.data)
            # Create a mask of where the max values are
            max_mask = self.data == np.broadcast_to(out.data, self.data.shape)
            grad += max_mask * out.grad
            self.grad += grad

        out._backward = _backward
        return out

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.data.astype(int)
        elif isinstance(key, tuple):
            key = tuple(k.data.astype(int) if isinstance(k, Tensor) else k for k in key)
        out_data = self.data[key]
        out = Tensor(out_data, children=[self], op='getitem')

        def _backward():
            # print('getitem backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = np.zeros_like(self.data)
            np.add.at(grad, key, out.grad)
            self.grad += grad

        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, children=[self], op='transpose')

        def _backward():
            # print('transpose backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def backward(self):
        if not self.requires_grad:
            return     
        nodes = []
        visited = set()

        def build_graph(node):
            if node not in visited and node.requires_grad:
                visited.add(node)
                for child in node.children:
                    build_graph(child)
                nodes.append(node)

        build_graph(self)
        for node in nodes:
            node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)
        for node in reversed(nodes):
            node._backward()
    
    def softmax(self, axis=-1):
        # Shift the logits for numerical stability
        shifted_logits = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_shifted = np.exp(shifted_logits)
        probs = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
        out = Tensor(probs, children=[self], op='softmax')

        def _backward():
            # print('softmax backward called')
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            grad = np.zeros_like(self.data)
            for i in range(len(self.data)):
                yi = out.data[i].reshape(-1, 1)
                dyi = out.grad[i].reshape(-1, 1)
                jacobian = np.diagflat(yi) - np.dot(yi, yi.T)
                grad[i] = (jacobian @ dyi).flatten()
            self.grad += grad

        out._backward = _backward
        return out

    # Additional methods as needed

    @classmethod
    def zeros_like(cls, tensor):
        return cls(np.zeros_like(tensor.data))

    @classmethod
    def ones_like(cls, tensor):
        return cls(np.ones_like(tensor.data))

    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape))

    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape))

    @classmethod
    def randint(cls, low, high, shape):
        return cls(np.random.randint(low, high, shape))

    @classmethod
    def randn(cls, shape):
        return cls(np.random.randn(*shape))

    def __len__(self):
        return len(self.data)
    
    def view(self, *shape):
        """
        Reshapes the tensor to the specified shape.

        Usage:
            tensor.view(new_shape)
            tensor.view(dim1, dim2, ...)
        """
        # Handle the case where a single tuple is passed
        if len(shape) == 1 and isinstance(shape[0], tuple):
            new_shape = shape[0]
        else:
            new_shape = shape

        # Reshape the data
        new_data = self.data.reshape(*new_shape)
        out = Tensor(new_data, children=[self], op='view')

        def _backward():
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # Reshape the gradient from the output to the input's shape
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    # [Existing backward method, softmax, and other methods]
