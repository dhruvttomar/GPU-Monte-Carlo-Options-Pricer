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
    const long long N = 1LL << 24;


    mt19937_64 gen(42);
    normal_distribution<double> dist(0.0, 1.0); 


    double sum = 0; 
    double sum_sq = 0;


    auto start = chrono::high_resolution_clock::now();


    for (long long i = 0; i < N; i++){
        double z = dist(gen);
        double S_T = S0 * exp(((r-(sigma*sigma)/2) * T + sigma*sqrt(T) * z));
        double payoff = max(S_T-K, 0.0); 
        sum += payoff;
        sum_sq += (payoff*payoff);
    }



    auto end = chrono::high_resolution_clock::now();



    double mean = sum/N*exp(-r*T);
    double variance = ((sum_sq/(N))-mean*mean);
    double SE = sqrt(variance)/sqrt(N)*exp(-r*T);


    double ms = chrono::duration<double, milli>(end - start).count();
    printf("Mean: : %.4f, Variace : %.4f, SE %.4f, Time %.4f\n", mean, variance, SE, ms);

    return 0;
}