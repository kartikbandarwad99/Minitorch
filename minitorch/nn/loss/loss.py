from minitorch import Tensor
import numpy as np

class Loss:
    def __init__(self):
        pass
    def cross_entropy_loss(logits, targets):
        # Make sure targets is 1D array of indices
        if len(targets.data.shape) > 1:
            targets = Tensor(targets.data.flatten())
        # Get batch size and num_classes
        batch_size = logits.data.shape[0]
        
        # Compute softmax with numerical stability
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = (logits - max_logits).exp()
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Get probabilities for the target classes
        batch_indices = Tensor(np.arange(batch_size))
        target_probs = probs[batch_indices, targets]
        
        loss = -(target_probs.log().sum()) / Tensor(float(batch_size))
        # loss = Tensor(loss,requires_grad=_grad_enabled)
        return loss

    def mean_squared_error_loss(prediction: Tensor, target: Tensor) -> Tensor:
        # Ensure prediction and target have the same shape
        if prediction.data.shape != target.data.shape:
            try:
                # Reshape target to match prediction's shape
                target_broadcasted = target.data.reshape(prediction.data.shape)
            except ValueError:
                raise ValueError(f"Shape mismatch: prediction shape {prediction.data.shape} vs target shape {target.data.shape}")

            target_tensor = Tensor(target_broadcasted, requires_grad=False)
        else:
            target_tensor = target

        squared_diff = (prediction - target_tensor) * (prediction - target_tensor)

        mse = squared_diff.sum() / prediction.data.size
        return mse
