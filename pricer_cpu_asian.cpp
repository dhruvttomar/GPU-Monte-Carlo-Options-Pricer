%%writefile pricer_gpu_asian.cu
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
using namespace std; 

int main() {
    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;
    const int M = 252;  
    const long long N = 1LL << 20;
    const double dt = T / M;


    mt19937_64 gen(42);
    normal_distribution<double> dist(0.0, 1.0); 


    double sum = 0; 
    double sum_sq = 0;


    auto start = chrono::high_resolution_clock::now();


    for (long long i = 0; i < N; i++){
        double S = S0; 
        double price_total = 0; 
        for (int k = 0; k < M; k++) { 
            curandStatePhilox4_32_10_t state; 
            curand_init(seed, i, 0, &state);
            double z = curand_normal_double(&state);
            S = S * exp(((r-(sigma*sigma)/2) * dt + sigma*sqrt(dt) * z));
            price_total += S;
        }
        double payoff = fmax(price_total / M - K, 0.0);
        sum += payoff;
        sum_sq += (payoff*payoff);
    }



    auto end = chrono::high_resolution_clock::now();



    double mean = sum / N;
    double variance = sum_sq / N - mean * mean;
    double price = mean * exp(-r * T);
    double SE = sqrt(variance) / sqrt(N) * exp(-r * T);


    double ms = chrono::duration<double, milli>(end - start).count();
    printf("Asian Price: : %.4f, Variace : %.4f, SE %.4f, Time %.4f\n", price, variance, SE, ms);

    return 0;
}