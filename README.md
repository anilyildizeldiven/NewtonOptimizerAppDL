# NewtonOptimizedModel

## Overview

The `NewtonOptimizedModel` is a TensorFlow Keras custom model designed to implement Newton's optimization method for training neural networks. This approach leverages both the first and second derivatives of the loss function (gradient and Hessian, respectively) to more accurately adjust model parameters during training. 

## Hessian Regularization

The `regularize_hessian` method adds a small constant to the diagonal of the Hessian matrix to ensure it remains invertible. This step is critical for stabilizing the Hessian inversion process, a key component of Newton's method, especially in scenarios where the Hessian might be singular or nearly singular.

## Custom Training Logic

The `train_step` method overrides the default training logic to incorporate Newton's optimization method. Here's a breakdown of its process:

**Loss Calculation**: Uses a `GradientTape` to make predictions and compute the loss by comparing predictions against true labels.

**Gradient and Hessian Computation**:
   - For each layer, gradients of the loss function with respect to the layer's trainable variables are computed.
   - Based on a subsampling parameter, it decides whether to compute Hessians for a more precise update (Newton's method) or to proceed with gradient descent updates.
**Parameter Updates**:
   - **Newton's Method**: For layers selected for Hessian computation, the method flattens the Hessian, regularizes it, inverts it, and then calculates the update step by multiplying the inverted Hessian with the gradient.
   - **Gradient Descent**: For layers not selected for Hessian computation, updates are made directly using the gradients.

## Considerations on Batch Size

When running the `NewtonOptimizedModel`, using batches smaller than the full size of the training data introduces bias in Hessian calculation. This bias occurs because the Hessian, calculated per batch, represents the curvature of the loss landscape based on the subset of data in the batch rather than the entire dataset. We used ```batch_size = X_train.shape[0]``` as batchsize (as can be seen in the example).

## Deep Dive into the Logic

The model is designed to experiment with Newton's method in the context of deep learning, providing insights into how second-order optimization might improve or affect the training process. By allowing for a subsampling parameter that controls the mix of Newton's method and traditional gradient descent updates, the model offers a flexible framework for investigating the benefits and challenges of incorporating Hessian-based updates in neural network training.

## Running the Model

When using the model, consider the impact of the ```subsampling_parameter```as well as the ```regularization_strength``` on the effectiveness and accuracy of Newton's optimization method. It's recommended to experiment with different values  to find the optimal configuration for your specific problem.

## Performance

tbd
