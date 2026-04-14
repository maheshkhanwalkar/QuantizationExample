# QuantizationExample
Example implementation of linear quantization

## Background
Linear quantization is a technique to reduce model size by representing
model weights with fewer bits, while still allowing computations to be
performed in the original bit width.

## Example Code
The code in `main.cpp` implements linear quantization for a simple matrix
multiplication operation, quantizing FP32 to INT8. The user can enter
the dimensions of the matrix, which the program uses to build a matrix of
that size with random values and a vector to multiply against. It prints
out model size statistics and execution time.

```
Quantization Example
Enter matrix dimensions: 30000 50000
Matrix memory size: (FP32) 5722.05 MiB
Execution time: 8.37162 seconds.
Result vector (first 5 entries): [8.85649e+08, 3.74333e+08, 2.82651e+08, -2.72156e+08, 2.67059e+09]
Quantized matrix size: (INT8) 1430.51 MiB
Quantized Execution time: 1.61158 seconds.
Result vector (first 5 entries): [8.87291e+08, 3.74186e+08, 2.83418e+08, -2.71801e+08, 2.67612e+09]
```

## Implementation Details
The example program is implemented in C++ and computes a matrix multiplication.
There is no explicit optimization performed. We compiled the executable using
CMake release mode, which turns on compiler optimizations (as we can see the use of AVX instructions)
