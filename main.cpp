#include <iostream>
#include <random>
#include <chrono>

/*
 * Quantize a fp32 to int8
 */
static int8_t quantize(
    const float r,
    const float scale,
    const int8_t zero_point,
    const int8_t q_min,
    const int8_t q_max
) {
    const auto raw_value = static_cast<int8_t>(roundf(r / scale + static_cast<float>(zero_point)));
    return std::clamp(raw_value, q_min, q_max);
}

/*
 * Dequantize an int8 back to fp32
 */
static float dequantize(const int8_t q, const float scale, const int8_t zero_point) {
    return scale * static_cast<float>(q - zero_point);
}

/*
 * Compute Mv, where M is a mxn matrix and v is a nx1 vector. Both the matrix and
 * vector values are fp32.
 */
static float* mat_mul_fp32(const float* matrix, const float* input_vec, int m, int n) {
    auto* output_vec = new float[m];
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * input_vec[j];
        }
        output_vec[i] = sum;
    }
    return output_vec;
}

/*
 * Compute Mv, where M is a mxn matrix and v is a nx1 vector. The matrix values are
 * quantized as int8 while the vector values are fp32.
 */
static float* mat_mul_int8(const int8_t* matrix, const float* input_vec, int m, int n, float scale, int8_t zero_point) {
    auto* output_vec = new float[m];
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += dequantize(matrix[i * n + j], scale, zero_point) * input_vec[j];
        }
        output_vec[i] = sum;
    }

    return output_vec;
}

static std::pair<float*, float*> allocate_matrix_and_input_vec(int m, int n) {
    auto* matrix = new float[n*m];
    std::random_device rd;
    std::mt19937 gen(rd());

    /**
     * Normalized weight values
     */
    std::uniform_real_distribution dist(0.0f, 1.0f);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n + j] = dist(gen);
        }
    }

    auto *input_vec = new float[n];

    for (int i = 0; i < n; i++) {
        input_vec[i] = dist(gen);
    }

    return std::make_pair(matrix, input_vec);
}

static int8_t* quantize_matrix(const float* matrix, int m, int n, float* scale, int8_t* zero_point) {
    float r_max = matrix[0];
    float r_min = matrix[0];

    for (int i = 1; i < m*n; i++) {
        if (matrix[i] > r_max) {
            r_max = matrix[i];
        }
        if (matrix[i] < r_min) {
            r_min = matrix[i];
        }
    }

    /*
     * Compute the scale and zero point values using the r_max,
     * r_min, q_max, q_min values of the weights.
     */
    int8_t q_max = INT8_MAX;
    int8_t q_min = INT8_MIN;

    *scale = (r_max - r_min) / static_cast<float>(q_max - q_min);
    *zero_point = static_cast<int8_t>(
        std::clamp(
            static_cast<int32_t>(roundf(static_cast<float>(q_min) - r_min / *scale)),
            static_cast<int32_t>(q_min),
            static_cast<int32_t>(q_max)
        )
    );

    auto* quantized_matrix = new int8_t[m*n];

    /*
     * Build out the quantized matrix
     */
    for (int i = 0; i < m*n; i++) {
        quantized_matrix[i] = quantize(matrix[i], *scale, *zero_point, q_min, q_max);
    }

    return quantized_matrix;
}

float loss_function(const float* result, const float* quantized_res, int m) {
    float loss = 0.0f;
    for (int i = 0; i < m; i++) {
        loss += powf(result[i] - quantized_res[i], 2);
    }
    return loss / static_cast<float>(m);
}

void print_result(const float* result) {
    std::cout << "Result vector (first 5 entries): ";
    std::cout << "[";
    for (int i = 0; i < 5; i++) {
        std::cout << result[i];
        if (i != 4) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

int main() {
    int m, n;

    std::cout << "Quantization Example" << std::endl;
    std::cout << "Enter matrix dimensions: ";
    std::cin >> m >> n;

    std::cout << "Matrix memory size: (FP32) " << 1.0 * n * m * sizeof(float) / 0x100000 << " MiB" << std::endl;

    auto [matrix, input_vec] = allocate_matrix_and_input_vec(m, n);

    auto start = std::chrono::high_resolution_clock::now();
    const float *result = mat_mul_fp32(matrix, input_vec, m, n);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds." << std::endl;

    print_result(result);

    float scale;
    int8_t zero_point;

    std::cout << "Quantized matrix size: (INT8) " << 1.0 * n * m * sizeof(int8_t) / 0x100000 << " MiB" << std::endl;

    auto quantized_matrix = quantize_matrix(matrix, m, n, &scale, &zero_point);

    start = std::chrono::high_resolution_clock::now();
    const float *quant_res = mat_mul_int8(quantized_matrix, input_vec, m, n, scale, zero_point);
    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;
    std::cout << "Quantized Execution time: " << elapsed.count() << " seconds." << std::endl;

    print_result(quant_res);

    std::cout << "Loss: " << loss_function(result, quant_res, m) << std::endl;

    delete[] quant_res;
    delete[] quantized_matrix;
    delete[] result;
    delete[] matrix;
    delete[] input_vec;

    return 0;
}
