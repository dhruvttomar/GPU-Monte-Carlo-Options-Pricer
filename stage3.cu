%%writefile pricer_gpu.cu
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

    double payoff = 0; 
    curandStatePhilox4_32_10_t state; 
    curand_init(seed, i, 0, &state);
    
    double my_sum = 0; 
    double my_sum_sq = 0; 
    for (int p = i; p < N; p += gridDim.x * blockDim.x) {
        double z = curand_normal_double(&state);
        double S_T = S0 * exp(((r-(sigma*sigma)/2) * T + sigma*sqrt(T) * z));
        payoff = fmax(S_T-K, 0.0); 
        my_sum += payoff;
        my_sum_sq += payoff*payoff; 
    }

    tile[t] = my_sum; 
    tile_sq[t] = my_sum_sq; 
    __syncthreads();
    for (int stride = 128; stride> 16; stride/= 2){
        if (t < stride){
            tile[t] += tile[t + stride];
            tile_sq[t] += tile_sq[t + stride];
        }
        __syncthreads();
    }

    if (t < 32) {
        double s  = tile[t];
        double s2 = tile_sq[t];
        for (int offset = 16; offset > 0; offset /= 2) {
            s += __shfl_down_sync(0xffffffff, s,  offset);
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
    int block_count = 157;

    size_t part_bytes = block_count * sizeof(double);

    double* d_partials;
    double* d_partials_sq;
    cudaMalloc(&d_partials, part_bytes);
    cudaMalloc(&d_partials_sq, part_bytes);

    price_kernel<<<block_count, 256>>>(d_partials, d_partials_sq, N, 42, S0, K, r, sigma, T);
    cudaDeviceSynchronize();
    auto start = chrono::high_resolution_clock::now();
    price_kernel<<<block_count, 256>>>(d_partials, d_partials_sq, N, 42, S0, K, r, sigma, T);    
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();

    double* h_partials = new double[block_count];
    double* h_partials_sq = new double[block_count];

    auto start2 = chrono::high_resolution_clock::now();
    cudaMemcpy(h_partials, d_partials, part_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partials_sq, d_partials_sq, part_bytes, cudaMemcpyDeviceToHost);

    double sum = 0;
    double sum_sq = 0;
    for (int b = 0; b < block_count; b++) {
        sum += h_partials[b];
        sum_sq += h_partials_sq[b];
    }
    auto end2 = chrono::high_resolution_clock::now();
    double ms2 = chrono::duration<double, milli>(end2 - start2).count();

    double mean = sum / N;
    double variance = sum_sq / N - mean * mean;
    double price = mean * exp(-r * T);
    double SE = sqrt(variance) / sqrt(N) * exp(-r * T);

    printf("Price %.4f, SE %.4f, kernel %.4f ms, copy+sum %.4f ms\n", price, SE, ms, ms2);

    cudaFree(d_partials);
    cudaFree(d_partials_sq);
    delete[] h_partials;
    delete[] h_partials_sq;
    return 0;
}