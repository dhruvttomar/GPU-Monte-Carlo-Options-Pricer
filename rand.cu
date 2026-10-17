#include <curand_kernel.h>

__global__ void fill_random(double* d_out, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, i, 0, &state);
        d_out[i] = curand_normal_double(&state);
    }
}