# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Daniel Gwon's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/dygwon.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Estimate a Scalar Derivative | Estimate a scalar polynomial derivative with a forward finite difference using coefficients ordered by ascending power. | https://www.tensortonic.com/problems/autograd-l01-finite-difference-derivative |
| Measure Scalar Expression Partials | Estimate the three partial derivatives of a scalar expression by perturbing one input at a time. | https://www.tensortonic.com/problems/autograd-l01-scalar-expression-partials |
| Implement Cosine Similarity | Implement cosine similarity from scratch with NumPy. Covers dot products, Euclidean norms and zero-vector handling, with tests in the browser. | https://www.tensortonic.com/problems/cosine-similarity |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/problems/cuda-leaky-relu |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Aggregation Functions | Compute selected NumPy aggregation functions globally or along a requested axis using float64 values. | https://www.tensortonic.com/problems/numpy-aggregation |
| Angle Features | Return a float64 array where row 0 contains the sine values, row 1 the cosine values, and row 2 the tangent values. | https://www.tensortonic.com/problems/numpy-angle-features |
| Arange and Linspace | Generate a one-dimensional NumPy sequence using either step-based arange or count-based linspace semantics. | https://www.tensortonic.com/problems/numpy-arange-linspace |
| Basic Indexing | Extract a rectangular NumPy subarray with row and column slice boundaries using standard basic indexing. | https://www.tensortonic.com/problems/numpy-basic-indexing |
| Boolean Masking | Build three filtered views of a 2D array: an element-level boolean mask, rows kept when any element exceeds a threshold. | https://www.tensortonic.com/problems/numpy-boolean-masking |
| Column Scaling | Scale every column of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/problems/numpy-col-scaling |
| Create Arrays from Lists | Convert a rectangular Python list of lists into a two-dimensional NumPy float64 array. | https://www.tensortonic.com/problems/numpy-create-array |
| Fancy Indexing | Convert the data to float64 and return the array formed by selecting elements along that axis using integer array indexing. | https://www.tensortonic.com/problems/numpy-fancy-indexing |
| Filter and Extract | Implement Filter and Extract, and apply a boolean mask to select values strictly greater than threshold. | https://www.tensortonic.com/problems/numpy-filter-extract |
| Mutation Trap | Extract an independent NumPy row copy, mutate it safely, and verify that the original array remains unchanged. | https://www.tensortonic.com/problems/numpy-mutation-trap |
| Normalized Difference | Use two 2D arrays a and b of the same shape and a scalar range [lo, hi], clip both arrays to [lo, hi], rescale each to [0, 1]. | https://www.tensortonic.com/problems/numpy-norm-diff |
| Norm-Gated Linear Transform | Compute the linear transform Z = X @ W, then zero out every row of Z whose L2 norm is strictly below the threshold. | https://www.tensortonic.com/problems/numpy-norm-gate |
| Normalize Columns | Standardize each NumPy matrix column by subtracting its mean and dividing by its population standard deviation. | https://www.tensortonic.com/problems/numpy-normalize-columns |
| Outer Sum | Compute the broadcasted outer sum of two NumPy vectors without loops, supporting different lengths and numeric values. | https://www.tensortonic.com/problems/numpy-outer-sum |
| Pairwise Differences | Implement Pairwise Differences, and compute the pairwise difference matrix without any Python loops. | https://www.tensortonic.com/problems/numpy-pairwise-diff |
| Random Array Generation | Generate seeded float64 NumPy arrays from either a uniform or standard normal distribution. | https://www.tensortonic.com/problems/numpy-random-arrays |
| Reshaping Arrays | Transform a rectangular input into a NumPy float64 array using flattening, transposition, or a leading batch dimension. | https://www.tensortonic.com/problems/numpy-reshape |
| Row Extremes | Implement Row Extremes, using np.argmax(axis=1) to find the column index of the maximum value in each row. | https://www.tensortonic.com/problems/numpy-row-extremes |
| Row Scaling | Scale every row of a NumPy matrix by its aligned weight through broadcasting, without explicit Python loops. | https://www.tensortonic.com/problems/numpy-row-scaling |
| Sort and Argsort | Return NumPy values sorted along a selected axis together with the indices that produce the same ordering. | https://www.tensortonic.com/problems/numpy-sort-argsort |
| Zeros and Ones | Create a two-dimensional float64 NumPy array of a requested shape filled entirely with zeros or ones. | https://www.tensortonic.com/problems/numpy-zeros-ones |
| Activation Functions | Implement four common activation functions from scratch using basic PyTorch tensor operations (no torch.nn module). | https://www.tensortonic.com/problems/pytorch-activation-function-from-scratch |
| Basic Autograd | Use PyTorch autograd to evaluate a scalar function and return its derivative at every supplied input value. | https://www.tensortonic.com/problems/pytorch-basic-autograd |
| Batch Normalization | Normalize each feature across the batch, then scale and shift using learnable parameters. | https://www.tensortonic.com/problems/pytorch-batch-normalization |
| Simple Neural Network | Implement a class SimpleNet subclassing nn.Module with two linear layers and ReLU between them. | https://www.tensortonic.com/problems/pytorch-build-simple-nn-from-scratch |
| Custom Linear Layer | Implement a custom linear layer that computes the affine transformation without using any built-in linear layer. | https://www.tensortonic.com/problems/pytorch-custom-linear-layer |
| Dropout from Scratch | Implement PyTorch inverted dropout from a supplied mask during training while returning inputs unchanged in evaluation mode. | https://www.tensortonic.com/problems/pytorch-dropout-from-scratch |
| Loss Functions | Implement three common loss functions from scratch using PyTorch tensor operations: mean squared error, cross-entropy, and Huber loss. | https://www.tensortonic.com/problems/pytorch-loss-functions |
| Softmax from Scratch | Implement numerically stable batched softmax in PyTorch by shifting logits before exponentiation and normalization. | https://www.tensortonic.com/problems/pytorch-softmax-from-scratch |
| Tensor Operations | Perform common element-wise and matrix tensor operations: add, multiply, matmul, power, and max. | https://www.tensortonic.com/problems/pytorch-tensor-arithmetic |
| Tensor Factory | Create PyTorch tensors with zeros, ones, or a constant fill value using the requested shape and dtype. | https://www.tensortonic.com/problems/pytorch-tensor-creation |
| Tensor Shape Manipulation | Reshape tensors using three common PyTorch operations: flatten to collapse into 1D, squeeze to remove size-1 dimensions. | https://www.tensortonic.com/problems/pytorch-tensor-reshape |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/problems/relu |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/problems/sigmoid |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/problems/softmax |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/problems/tanh |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-subtract |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/dygwon)
<!-- tensortonic:end -->
