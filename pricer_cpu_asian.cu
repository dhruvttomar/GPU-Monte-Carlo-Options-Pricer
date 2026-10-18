%%writefile pricer_gpu_asian.cu
#include <cstdio>
#include <cmath>
#include <chrono>
#include <curand_kernel.h>
using namespace std;

__global__ void price_kernel(double* d_partials, double* d_partials_sq, int N,
                             unsigned long long seed,
                             double S0, double K, double r, double sigma, double T) {
    __shared__ double tile[256];
    __shared__ double tile_sq[256];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);       

    const int M = 252;
    double dt = T / M;
    double my_sum = 0;
    double my_sum_sq = 0;

    for (int p = i; p < N; p += gridDim.x * blockDim.x) {   
        double S_a = S0;
        double S_b = S0;
        double total_a = 0;
        double total_b = 0;
        for (int k = 0; k < M; k++) { 
            double z = curand_normal_double(&state);
            S_a = S_a * exp((r - sigma*sigma/2) * dt + sigma*sqrt(dt) * z);
            S_b = S_b * exp((r - sigma*sigma/2) * dt + sigma*sqrt(dt) * -z);
            total_a += S_a;
            total_b += S_b;
        }

        double payoff_a = fmax(total_a / M - K, 0.0);
        double payoff_b = fmax(total_b / M - K, 0.0);
        double pair = (payoff_b + payoff_a)/2;
        my_sum += pair;
        my_sum_sq += pair * pair;
    }

    tile[t] = my_sum;
    tile_sq[t] = my_sum_sq;
    __syncthreads();

    for (int stride = 128; stride >= 32; stride /= 2) {
        if (t < stride) {
            tile[t] += tile[t + stride];
            tile_sq[t] += tile_sq[t + stride];
        }
        __syncthreads();
    }

    if (t < 32) {
        double s  = tile[t];
        double s2 = tile_sq[t];
        for (int offset = 16; offset > 0; offset /= 2) {
            s  += __shfl_down_sync(0xffffffff, s,  offset);
            s2 += __shfl_down_sync(0xffffffff, s2, offset);
        }
        if (t == 0) {
            d_partials[blockIdx.x] = s;
            d_partials_sq[blockIdx.x] = s2;
        }
    }
}

int main() {
    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;
    const int N = 1 << 20;
    int block_count = 160;

    size_t part_bytes = block_count * sizeof(double);
    double* d_partials;
    double* d_partials_sq;
    cudaMalloc(&d_partials, part_bytes);
    cudaMalloc(&d_partials_sq, part_bytes);

    price_kernel<<<block_count, 256>>>(d_partials, d_partials_sq, N/2, 42, S0, K, r, sigma, T);
    cudaDeviceSynchronize();

    auto start = chrono::high_resolution_clock::now();
    price_kernel<<<block_count, 256>>>(d_partials, d_partials_sq, N/2, 42, S0, K, r, sigma, T);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();

    double* h_partials = new double[block_count];
    double* h_partials_sq = new double[block_count];
    cudaMemcpy(h_partials, d_partials, part_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partials_sq, d_partials_sq, part_bytes, cudaMemcpyDeviceToHost);

    double sum = 0, sum_sq = 0;
    for (int b = 0; b < block_count; b++) {
        sum += h_partials[b];
        sum_sq += h_partials_sq[b];
    }

    double mean = sum / (N/2);
    double variance = sum_sq /(N/2) - mean * mean;
    double price = mean * exp(-r * T);
    double SE = sqrt(variance) / sqrt(N) * exp(-r * T);

    printf("Asian price %.4f, SE %.4f, kernel %.4f ms\n", price, SE, ms);

    cudaFree(d_partials);
    cudaFree(d_partials_sq);
    delete[] h_partials;
    delete[] h_partials_sq;
    return 0;
}