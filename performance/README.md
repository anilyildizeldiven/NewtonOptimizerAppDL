## Performance Tests of the Subsampled Newton Optimizer

We conducted a series of performance tests for our custom model class, leveraging the Subsampled Newton Optimizer, across three datasets from the UCI database:
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)
- [Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile)

### Configuration
For our tests, we configured the custom model with the following parameters:
- **Regularization Strength**: `1e-4`
- **Subsampling Parameter**: `0.5`

We compared the performance of our model against two commonly used optimizers:
- **SGD (Stochastic Gradient Descent)**
- **Adam**

Both optimizers were set with a `batch_size` of `32` and a `learning_rate` of `0.01`.

### Testing Procedure
Each optimizer, including our custom Subsampled Newton Optimizer, was run for `100` epochs across `10` runs. The results are presented as plots showing the average loss convergence for each configuration.

### Neural Network Architectures
The tests were conducted on neural networks with varying depths to assess the adaptability and efficiency of the Subsampled Newton Optimizer:
- **One-Layer Net**: A simple network with a single hidden layer.
- **Five-Layer Net**: A more complex network with five hidden layers.
- **Twenty-Layer Net**: A deep network architecture with twenty hidden layers.

### Results
The performance plots demonstrate how our custom Subsampled Newton Optimizer compares to the SGD and Adam optimizers in terms of loss convergence across the different datasets and network architectures. These results highlight the effectiveness and robustness of our optimizer in various scenarios.

---

This structured format provides a clear and concise overview of your performance tests, making it easy for readers to understand the scope, methodology, and outcomes of your experiments.
