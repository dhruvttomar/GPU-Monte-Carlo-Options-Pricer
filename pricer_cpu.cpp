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
    const long long N = 1LL << 20;

    // TODO: set up the random generator and normal distribution

    mt19937_64 gen(42);
    normal_distribution<double> dist(0.0, 1.0); 

    // TODO: running totals

    double sum = 0; 
    double sum_sq = 0;

    // TODO: start timer

    auto start = chrono::high_resolution_clock::now();

    // TODO: loop over N paths
    //       draw Z, compute S_T, compute payoff, update totals

    for (long long i = 0; i < N; i++){
        double z = dist(gen);
        //S_T = S0 × e^((r − σ²/2)T + σ√T × Z)
        double S_T = S0 * exp(((r-(sigma*sigma)/2) * T + sigma*sqrt(T) * z));
        double payoff = max(S_T-K, 0.0); 
        sum += payoff;
        sum_sq += (payoff*payoff);
    }


    // TODO: stop timer

    auto end = chrono::high_resolution_clock::now();


    // TODO: compute mean, variance, SE; discount price and SE

    double mean = sum/N*exp(-r*T);
    double variance = ((sum_sq/(N))-mean*mean);
    double SE = sqrt(variance)/sqrt(N)*exp(-r*T);

    // TODO: prdouble price, SE, time in ms

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Mean: : %.4f, Variace : %.4f, SE %.4f, Time %.4f\n", mean, variance, SE, ms);

    return 0;
}