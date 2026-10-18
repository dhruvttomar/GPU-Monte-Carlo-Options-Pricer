%%writefile pricer_gpu.cu
#include <cstdio>
#include <cmath>
#include <chrono>
#include <curand_kernel.h>

using namespace std; 

__global__ void price_kernel(double* d_payoffs, int N, unsigned long long seed,
                             double S0, double K, double r, double sigma, double T) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        curandStatePhilox4_32_10_t state;  
        curand_init(seed, i, 0, &state);
        double z = curand_normal_double(&state);
        double S_T = S0 * exp(((r-(sigma*sigma)/2) * T + sigma*sqrt(T) * z));
        double payoff = fmax(S_T-K, 0.0); 
        d_payoffs[i] = payoff;
    }
}

int main() {
    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;
    const int N = 1 << 20;


    size_t bytes = N * sizeof(double);
    double* d_payoffs;
    cudaMalloc(&d_payoffs, bytes);  

    
 


    auto start = chrono::high_resolution_clock::now();
    int block_count = (N + 255) / 256;
    price_kernel<<<block_count, 256>>>(d_payoffs, N, 42, S0, K, r, sigma, T);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(end - start).count();



    double* h_payoffs = new double[N];
    
    auto start2 = chrono::high_resolution_clock::now();
    cudaMemcpy(h_payoffs, d_payoffs, bytes, cudaMemcpyDeviceToHost); 


    double sum = 0; 
    double sum_sq = 0; 

    for (long long i = 0; i < N; i++){
        sum += h_payoffs[i];
        sum_sq += (h_payoffs[i] * h_payoffs[i]);
    }
    
    auto end2 = chrono::high_resolution_clock::now();
    double ms2 = chrono::duration<double, milli>(end2 - start2).count();

    double mean = sum/N;
    double variance = ((sum_sq/(N))-mean*mean);
    double price = sum/N*exp(-r*T);
    double SE = sqrt(variance)/sqrt(N)*exp(-r*T);

    printf("Price: : %.4f, Variance : %.4f, SE %.4f, Time %.4f\n", price, variance, SE, ms+ms2);

    cudaFree(d_payoffs); 
    delete[] h_payoffs; 

    return 0;
}