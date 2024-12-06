# Minitorch Library using Numpy
A from-scratch implementation of PyTorch's core functionality using NumPy. This project aims to help understand the internals of deep learning frameworks by building key components from the ground up. 

## 📋 Table of Contents
- [🌟 Features](#-features)
- [📁 Folder Structure](#-folder-structure)
- [🛠️ Installation](#-installation)
- [🚀 Getting Started](#-getting-started)
  - [Creating Tensors](#creating-tensors)
  - [Building Models](#building-models)
  - [Training a Model](#training-a-model)
- [⚠️ Issues Faced](#-issues-faced)
- [🚧 Work in Progress](#-work-in-progress)
- [📄 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
- [💬 Feedback](#-feedback)

## 🌟 Features

* Tensor Operations: Support for basic tensor operations with back propogation.
* Neural Network Modules: Implementation of common layers like Linear, Tanh, Softmax, Embedding, and Flatten.
* Sequential Models: Easy model building using the Sequential container.
* Optimizers: Stochastic Gradient Descent (SGD) optimizer for training models.
* Loss Functions: Mean Squared Error (MSE) and Cross-Entropy loss functions.

## 📁 Folder Structure

```
minitorch
├── Tensor
│   ├── __init__.py
│   └── tensor.py
└── nn
    ├── __init__.py
    ├── modules
    │   ├── __init__.py
    │   ├── linear.py
    │   ├── tanh.py
    │   ├── softmax.py
    │   ├── embedding.py
    │   └── flatten.py
    └── models
    │   ├── __init__.py
    │   └── sequential.py
    ├── optim
    │   ├── __init__.py
    │   └── sgd.py
    └── loss
        ├── __init__.py
        └── loss.py
```

* Tensor: Core tensor class with support for back propogation.
* nn: Neural network modules, models, optimizers, and loss functions.
    * modules: Individual layer implementations
    * models: Model containers like Sequential.
    * optim: Optimizers for training.
    * loss: Loss function implementations.

## 🛠️ Installation

### ⚙️ Prerequisites
* Python 3.6 or higher
* NumPy library
### 📥 Steps
1. **Clone the Repository**

    ```bash 
    git clone https://github.com/yourusername/minitorch.git 
    ```

2. **Navigate to the Project Directory**
    ```bash 
    cd minitorch
    ``` 

3. **Install Dependencies**
    ```bash 
    pip install -r requirements.txt
    ``` 

4. **Install Minitorch**
You can install Minitorch in editable mode using pip:
    ```bash 
    pip install -e .
    ``` 
## 🚀 Getting Started

### Creating Tensors
```python
import minitorch as torch

x = torch.Tensor([1.0, 2.0, 3.0])

# Basic operations
y = x * 2
z = y.sum()

# Backward pass
z.backward()

# Gradients
print("Gradients:", x.grad)  # Output: [2.0, 2.0, 2.0]
``` 
### Building Models
You can build neural network models using the provided modules.
```python
import minitorch 
from minitorch.nn import Linear, Tanh, Sequential

# Define a simple model
model = Sequential([
    Linear(in_features=3, out_features=5),
    Tanh(),
    Linear(in_features=5, out_features=1)
])
```
### Training a Model
Here's an example of training a simple classification model using the cross entropy loss and SGD optimizer.
```python
import minitorch
from minitorch import Tensor
from minitorch.nn import Sequential,Linear,Tanh,Embedding,Flatten,Loss
from minitorch.nn.optim import SGD

# Assume that the classification problem has 5 classes
X = Tensor.randn((10,10))
Y = Tensor.randint(0,5,(10))

model = Sequential([Linear((10,100)),
                    Tanh(),
                    Linear((100,5))
                    ])

learning_rate = 0.1
optimizer = SGD(model.parameters(), lr=learning_rate)
num_epochs = 1000
batch_size = 64

for epoch in range(num_epochs):
    indexes = Tensor.randint(0, len(X), (batch_size,))
    x_mini = X[indexes]
    y_mini = Y[indexes]

    # Forward pass
    out = model(x_mini)

    # Compute MSE loss
    mse = Loss.mean_squared_error_loss(out, y_mini)
    errors.append(mse.data.item())
    # Print loss every 10 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: MSE Loss = {mse.data}')

    # Backward pass
    mse.backward()

    # Update parameters using the optimizer
    optimizer.step()

    # To check if the backpropogation is working uncomment and run the below code and check if the gradients are not none
    # for p in model.parameters():
    #     print(p.grad)

    # Reset gradients
    optimizer.zero_grad()
```
**Output**
```python
0: Tensor:1.7039032720417477
100: Tensor:1.1616989206076886
200: Tensor:1.0857209949164703
300: Tensor:0.6746197129345434
400: Tensor:0.5034095525703973
500: Tensor:0.5123857403368544
600: Tensor:0.2778792183180653
700: Tensor:0.18678039362325244
800: Tensor:0.08656333425684432
900: Tensor:0.1036119706235993
999: Tensor:0.08642679899338912

```

📝 **Note:** In the PyTorch library, the `randn` and `randint` functions are called as `torch.randn` and `torch.randint`. However, in Minitorch, both of these functions have been integrated into the `Tensor` class itself. Hence, they are called as `Tensor.randn` and `Tensor.randint`.

## ⚠️ Issues Faced

During the development of Minitorch, several challenges were encountered, particularly related to broadcasting, differentiation, and backpropagation. Below are the key issues and their descriptions:

### 🔄 Broadcasting
**Shape Mismatches**: Operations between tensors of incompatible shapes (e.g., (32, 1) vs (32,)) led to errors during element-wise computations.

**Solution**: Implemented shape alignment within loss functions and introduced helper functions to ensure consistent tensor dimensions across operations.

### 🔁 Differentiation
**Gradient Accumulation**: In certain operations, gradients were not being correctly accumulated, leading to incorrect parameter updates.

**Solution**: Reviewed and corrected the backward pass implementations in tensor operations to ensure accurate gradient propagation.

### ↩️ Backpropagation
**Exploding Gradients**: High learning rates and random normal weight initialization in Linear layers caused gradients to explode, resulting in loss values reaching inf or nan.
Solution: Reduced the learning rate and implemented proper weight initialization by multiplying weights with a scaling factor to stabilize training.

## 🚧 Work in Progress

Here are the features I am currently working on:

- **Implementing Popular Optimizers**: Adding optimizers such as Adam and AdamW.
- **`no_grad()` Functionality**: Enabling computations without maintaining the computation graph.
- **Convolutional Layers**: Extending the module to include convolutional layers.
- **Higher-Dimensional Tensors**: Enhancing the `Tensor` class to efficiently handle tensors with more than two dimensions.
- **Batch Normalization Layers**: Implementing batch normalization layers instead of merely initializing weights with scaling factors.
- **`requires_grad` option: Adding the requires_grad option for tensors. In Minitorch, tensors have gradients enabled by default, and this feature will allow users to control whether gradients are tracked for specific tensors.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Minitorch is inspired by PyTorch and the desire to understand how the PyTorch library works in the backend. It is designed and structured similarly to the PyTorch library, but there are certain areas where this repository differs from actual PyTorch to keep it simple. You can refer to the examples for the code. Additionally, Andrej Karpathy's lectures have been instrumental in guiding the development of this project.

## 💬 Feedback
For questions or discussions, please reach out to me at kbandarwad@gmail.com.
