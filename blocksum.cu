__global__ void price_kernel(double* d_partials, double* d_partials_sq, int N,
                             unsigned long long seed,
                             double S0, double K, double r, double sigma, double T) {
    __shared__ double tile[256];
    __shared__ double tile_sq[256];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;

    double payoff = 0; 

    if (i < N){
        curandStatePhilox4_32_10_t state;  
        curand_init(seed, i, 0, &state);
        double z = curand_normal_double(&state);
        double S_T = S0 * exp(((r-(sigma*sigma)/2) * T + sigma*sqrt(T) * z));
        payoff = fmax(S_T-K, 0.0); 
    } else {
        payoff = 0;
    }
    tile[t] = payoff; 
    tile_sq[t] = payoff*payoff; 
    __syncthreads();
    for (int stride = 256; stride> 0; stride/= 2){
        if (t < stride){
            tile[t] += tile[t + stride];
            tile_sq[t] += tile_sq[t + stride];
        }
        __syncthreads();
    }

    if (t == 0){
        d_partials[blockIdx.x] = tile[0];
        d_partials_sq[blockIdx.x] = tile_sq[0];
    }
}