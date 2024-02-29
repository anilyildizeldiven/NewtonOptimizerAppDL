# Newton Optimized Neural Network Model

This repository introduces the `SubsampledNewtonOptimizedModel`, a custom TensorFlow Keras model designed to incorporate Newton's method into the training process of neural networks. Unlike traditional gradient descent-based optimization methods commonly used in deep learning, this model attempts to leverage the second-order optimization technique provided by Newton's method, offering a novel approach to minimizing the loss function by considering the curvature of the loss landscape.

## Overview

Newton's method, a cornerstone of numerical optimization, utilizes the Hessian matrix (a square matrix of second-order partial derivatives of the function being optimized) along with the gradient to compute update steps. This method can potentially offer faster convergence to a minimum by taking into account the curvature of the loss function, a feature not utilized by first-order methods like SGD or Adam.

However, Newton's method is computationally intensive, primarily due to the need for computing and inverting the Hessian matrix, which can be prohibitive for high-dimensional problems typical in deep learning. The `NewtonOptimizedModel` addresses this challenge by introducing a novel approach that balances the computational efficiency with the benefits of second-order optimization.

## Model Architecture

### Forward Pass

The model defines a forward pass through its neural network architecture using the `call` method, sequentially passing input data through its layers to produce predictions.

### Hessian Regularization

To address the numerical instability issues associated with Hessian inversion, the model includes a `regularize_hessian` method. This method adds a small value to the diagonal of the Hessian matrix, ensuring it remains invertible.

### Custom Training Logic

The core innovation lies in the `train_step` method, which overrides the default training behavior to implement Newton's method for optimization:

- **Gradients and Hessians Computation**: For each layer, gradients are computed using TensorFlow's automatic differentiation. For layers selected for Newton's optimization (based on a subsampling parameter), the Hessians are also computed.
- **Subsampling Strategy**: Instead of applying Newton's method uniformly across all layers, the model employs a probabilistic approach to decide for each layer whether to update its variables using Newton's method or traditional gradient descent. This strategy aims to balance computational efficiency with the optimization benefits of Newton's method.
- **Parameter Updates**: For variables selected for Newton's optimization, the model computes update steps by inverting the regularized Hessian matrix and applying it to the gradient. For other variables, updates are made using a gradient descent approach.

## Usage

The repository includes a comprehensive guide to setting up and training the `NewtonOptimizedModel` with a sample dataset (e.g., Iris dataset). Users can customize the model architecture, adjust the subsampling parameter, and experiment with different configurations to explore the effectiveness of Newton's method in various scenarios.

## Caveats

Be cautious about the batch size when testing the Code. In order for the Hessians not to be biased use ```batch_size = X_train.shape[0]``` as batchsize (as can be seen in the example).
