# Subsampled Newton Method Optimized Model


The subsampled `NewtonOptimizedModel` is acustom TensorFlow model that uses the Newton Method to update the Variables of a Neural Net together with a subsampling rate.
The built in subsampling reduces the computational burden of second order updates.

The performance of this custom model has been meticulously plotted against common optimizers like SGD (Stochastic Gradient Descent) and Adam, showcasing its  capabilities. These results are accessible in the "Performance" folder for detailed analysis.

## How the Model Works

### Core Architecture

At its core, the `NewtonOptimizedModel` is built upon TensorFlow's `Model` class, incorporating layers such as `Dense` with activation functions and initializers tailored for specific tasks. One can customize the model's architecture, including the number of classes (`num_classes`) and the subsampling rate (`subsampling_rate`).

### Newton's Method Optimization

The key feature of this model is its implementation of the Newton optimization method which generally allows for faster convergence to the minimum of the loss function compared to first-order methods.

### Regularization and Hessian Matrix

To ensure stability and prevent inversion problems, the Hessian matrix is regularized by adding a small value (`regularization_strength`) to its diagonal. This step mitigates the issue of ill-conditioned matrices, which can arise due to the high dimensionality of deep learning models.

### Subsampling for Scalability

Given the computational complexity of calculating the Hessian matrix for all variables, the model implements a subsampling strategy controlled by the `subsampling_rate` parameter. This rate determines the fraction of variables for which the Hessian and its inverse will be computed. For non-sampled variables, updates are scaled using the average update norm derived from the subsampled variables and their gradients.

### Ensuring Update Compatibility

The model meticulously ensures that updates are compatible with the shapes of the respective variables. This is achieved by flattening the Hessian matrix, performing the necessary matrix operations, and then reshaping the update vector to match the variable's dimensions (as demonstrated in this simple example: https://www.tensorflow.org/guide/advanced_autodiff#example_hessian)

## Parameters:

- **`regularization_strength`**: A parameter that controls the degree of regularization applied to the Hessian matrix, preventing numerical instabilities during its inversion.
- **`subsampling_rate`**: This parameter dictates the proportion of variables selected for Hessian computation and Newton's method updates. A higher rate increases computation but may lead to faster convergence. 
