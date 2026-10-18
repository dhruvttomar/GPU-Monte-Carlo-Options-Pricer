import numpy as np
import pyopencl as cl
import time, math

KERNEL_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

ulong splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}
ulong next_u64(ulong* s) {
    ulong x = *s;
    x ^= x >> 12;  x ^= x << 25;  x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DUL;
}
double next_uniform(ulong* s) {
    return ((next_u64(s) >> 11) + 1) * (1.0 / 9007199254740992.0);
}
double next_normal(ulong* s) {
    double u1 = next_uniform(s);
    double u2 = next_uniform(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

__kernel void price_kernel(__global double* d_partials, __global double* d_partials_sq,
                           const int N_pairs, const ulong seed,
                           const double S0, const double K, const double r,
                           const double sigma, const double T) {
    __local double tile[256];
    __local double tile_sq[256];

    int t = get_local_id(0);
    int i = get_global_id(0);
    ulong state = splitmix64(seed ^ splitmix64((ulong)i));

    const int M = 252;
    double dt = T / M;
    double my_sum = 0.0;
    double my_sum_sq = 0.0;


    for (int p = i; p < N_pairs; p += get_global_size(0)) {
    double S_a = S0, S_b = S0;
    double total_a = 0.0, total_b = 0.0;
    for (int k = 0; k < M; k++) {
        double z = next_normal(&state);
        S_a = S_a * exp((r - sigma*sigma/2) * dt + sigma*sqrt(dt) * z);
        S_b = S_b * exp((r - sigma*sigma/2) * dt - sigma*sqrt(dt) * z);
        total_a += S_a;
        total_b += S_b;
    }
    double payoff_a = fmax(total_a / M - K, 0.0);
    double payoff_b = fmax(total_b / M - K, 0.0);
    double pair = (payoff_a + payoff_b) / 2.0;
    my_sum += pair;
    my_sum_sq += pair * pair;
    }


    tile[t] = my_sum;
    tile_sq[t] = my_sum_sq; 
    barrier(CLK_LOCAL_MEM_FENCE);


    for (int stride = 128; stride >= 1; stride /= 2) {
        if (t < stride) {
            tile[t] += tile[t + stride];
            tile_sq[t] += tile_sq[t + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (t == 0){
        d_partials[get_group_id(0)] = tile[0];
        d_partials_sq[get_group_id(0)] = tile_sq[0];
    }
}
"""

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 1 << 20
N_pairs = N // 2
local_size = 256
groups = 120
global_size = groups * local_size

device = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, KERNEL_SRC).build()
kernel = prg.price_kernel

h_partials = np.zeros(groups, dtype=np.float64)
h_partials_sq = np.zeros(groups, dtype=np.float64)
mf = cl.mem_flags
d_partials = cl.Buffer(ctx, mf.WRITE_ONLY, h_partials.nbytes)
d_partials_sq = cl.Buffer(ctx, mf.WRITE_ONLY, h_partials_sq.nbytes)

args = (d_partials, d_partials_sq, np.int32(N_pairs), np.uint64(42),
        np.float64(S0), np.float64(K), np.float64(r), np.float64(sigma), np.float64(T))

kernel(queue, (global_size,), (local_size,), *args)
queue.finish()

t0 = time.perf_counter()
kernel(queue, (global_size,), (local_size,), *args)
queue.finish()
ms = (time.perf_counter() - t0) * 1000

cl.enqueue_copy(queue, h_partials, d_partials)
cl.enqueue_copy(queue, h_partials_sq, d_partials_sq)
queue.finish()

s, s2 = h_partials.sum(), h_partials_sq.sum()
mean = s / N_pairs
variance = s2 / N_pairs - mean * mean
price = mean * math.exp(-r * T)
SE = math.sqrt(variance) / math.sqrt(N_pairs) * math.exp(-r * T)
print(f"OpenCL Asian price {price:.4f}, SE {SE:.4f}, kernel {ms:.2f} ms")