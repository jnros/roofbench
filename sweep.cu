/*
 * sweep.cu - sweep operational intensity across the roofline
 *
 * Each kernel loads a float4 (4 floats) from DRAM, runs FMAS
 * fused multiply-add passes over the 4 values, then stores back.
 *
 * FMA = fused multiply-add: computes a*b+c in one hardware instruction,
 * rounding only at the end.  Counts as 2 FLOPs (one multiply, one add)
 * but executes in one clock cycle on modern GPUs.
 *
 * Operational intensity = FLOPs / bytes transferred:
 *
 *   FLOPs per float4 = FMAS * 4 components * 2 FLOPs per FMA = 8*FMAS
 *   bytes per float4 = 16 read + 16 write = 32
 *   intensity        = 8*FMAS / 32 = FMAS/4   FLOP/byte
 *
 * so FMAS = intensity * 4.  The template is parameterized by FMAS
 * (an integer) so #pragma unroll emits a straight FMA sequence with
 * no loop overhead; intensity_table drives the sweep from the outside.
 *
 * Ridge (peak_FLOPS / peak_BW) ~17 FLOP/byte on this GPU, between
 * the 16 and 32 FLOP/byte rows.
 */
#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
	cudaError_t e = (x); \
	if (e != cudaSuccess) { \
		fprintf(stderr, "CUDA %s:%d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(1); \
	} \
} while (0)

/*
 * FMAS: number of FMA passes per loaded element.
 * #pragma unroll forces compile-time unrolling so the inner body
 * becomes a flat sequence of FMA instructions with no branch.
 */
template<int FMAS>
__global__ void kernel_sweep(const float4 * __restrict__ src,
			     float4 * __restrict__ dst,
			     long n)
{
	long tid    = (long)blockIdx.x * blockDim.x + threadIdx.x;
	long stride = (long)gridDim.x  * blockDim.x;
	const float c = 1.000001f;

	for (long i = tid; i < n; i += stride) {
		float4 v = src[i];
#pragma unroll
		for (int f = 0; f < FMAS; f++) {
			/* one FMA per component: v = v*c + c (2 FLOPs each) */
			v.x = v.x * c + c;
			v.y = v.y * c + c;
			v.z = v.z * c + c;
			v.w = v.w * c + c;
		}
		dst[i] = v;
	}
}

/* ------------------------------------------------------------------ */

typedef void (*kfn)(const float4*, float4*, long);

/*
 * intensity_table: FLOP/byte values to sweep.
 * kern_table: corresponding kernel instantiation (FMAS = intensity * 4).
 * Ridge ~17 FLOP/byte lands between the 16 and 32 entries.
 */
#define NPTS 11
static const double intensity_table[NPTS] = {
	0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256
};
static const kfn kern_table[NPTS] = {
	kernel_sweep<1>,   kernel_sweep<2>,   kernel_sweep<4>,
	kernel_sweep<8>,   kernel_sweep<16>,  kernel_sweep<32>,
	kernel_sweep<64>,  kernel_sweep<128>, kernel_sweep<256>,
	kernel_sweep<512>, kernel_sweep<1024>
};

/*
 * FP32 CUDA cores per SM varies by compute capability.
 * Source: CUDA SDK helper_cuda.h _ConvertSMVer2Cores().
 */
static int fp32_cores_per_sm(int major, int minor)
{
	static const struct { int cc, cores; } tbl[] = {
		{30, 192}, {32, 192}, {35, 192}, {37, 192}, /* Kepler  */
		{50, 128}, {52, 128}, {53, 128},             /* Maxwell */
		{60,  64}, {61, 128}, {62, 128},             /* Pascal  */
		{70,  64}, {72,  64},                        /* Volta   */
		{75,  64},                                   /* Turing  */
		{80,  64}, {86, 128}, {87, 128}, {89, 128}, /* Ampere/Ada */
		{90, 128},                                   /* Hopper  */
		{100,128},                                   /* Blackwell (tentative) */
	};
	int cc = major * 10 + minor;
	for (int i = 0; i < (int)(sizeof tbl / sizeof tbl[0]); i++)
		if (tbl[i].cc == cc)
			return tbl[i].cores;
	return -1; /* unknown architecture */
}

static double elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
	float ms;
	cudaEventElapsedTime(&ms, a, b);
	return ms;
}

int main(void)
{
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

	int sm_clock_khz, mem_clock_khz, mem_bus_width;
	CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz,
		cudaDevAttrClockRate, 0));
	CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz,
		cudaDevAttrMemoryClockRate, 0));
	CUDA_CHECK(cudaDeviceGetAttribute(&mem_bus_width,
		cudaDevAttrGlobalMemoryBusWidth, 0));

	double peak_bw    = 2.0 * mem_clock_khz * 1e3
			    * mem_bus_width / 8.0 / 1e9; /* GB/s */
	int cores_per_sm = fp32_cores_per_sm(prop.major, prop.minor);
	if (cores_per_sm < 0) {
		fprintf(stderr, "unknown compute capability %d.%d\n",
			prop.major, prop.minor);
		return 1;
	}
	double peak_flops = prop.multiProcessorCount * cores_per_sm * 2.0
			    * sm_clock_khz * 1e3 / 1e12;  /* TFLOPS */
	double ridge      = peak_flops * 1e12 / (peak_bw * 1e9); /* FLOP/byte */

	printf("Device: %s   SMs: %d   sm_%d%d   %d FP32/SM\n",
		prop.name, prop.multiProcessorCount,
		prop.major, prop.minor, cores_per_sm);
	printf("Peak FP32 (base clk): %.2f TFLOPS\n", peak_flops);
	printf("Peak BW:              %.1f GB/s\n", peak_bw);
	printf("Ridge (theoretical):  %.1f FLOP/byte\n\n", ridge);

	/* 128 MB — larger than L2 (4 MB) to force DRAM traffic */
	long   n     = (128L << 20) / sizeof(float4);
	size_t bytes = n * sizeof(float4);

	float4 *d_src, *d_dst;
	CUDA_CHECK(cudaMalloc(&d_src, bytes));
	CUDA_CHECK(cudaMalloc(&d_dst, bytes));
	CUDA_CHECK(cudaMemset(d_src, 0, bytes));

	int threads = 256;
	int blocks  = prop.multiProcessorCount * 8;

	cudaEvent_t t0, t1;
	CUDA_CHECK(cudaEventCreate(&t0));
	CUDA_CHECK(cudaEventCreate(&t1));

	printf("%-10s  %-10s  %-10s  %-8s  %s\n",
		"FLOP/byte", "GFLOPS", "BW(GB/s)", "time(ms)", "bound");
	printf("%-10s  %-10s  %-10s  %-8s  %s\n",
		"----------", "----------", "----------", "--------", "-----");

	for (int k = 0; k < NPTS; k++) {
		double intensity = intensity_table[k];
		kfn    kern      = kern_table[k];

		/* warmup */
		kern<<<blocks, threads>>>(d_src, d_dst, n);
		CUDA_CHECK(cudaDeviceSynchronize());

		/* best of 3 runs */
		double best_ms = DBL_MAX;
		for (int run = 0; run < 3; run++) {
			CUDA_CHECK(cudaEventRecord(t0));
			kern<<<blocks, threads>>>(d_src, d_dst, n);
			CUDA_CHECK(cudaEventRecord(t1));
			CUDA_CHECK(cudaDeviceSynchronize());
			double ms = elapsed_ms(t0, t1);
			if (ms < best_ms)
				best_ms = ms;
		}

		/* intensity = FLOPs / bytes, so FLOPs = intensity * bytes */
		double flops  = intensity * 32.0 * n; /* 32 bytes per float4 */
		double gflops = flops / best_ms / 1e6;
		double gbytes = 2.0 * bytes / best_ms / 1e6; /* read + write */

		const char *bound = (intensity < ridge) ? "mem" : "compute";

		printf("%-10.2f  %-10.1f  %-10.1f  %-8.2f  %s\n",
			intensity, gflops, gbytes, best_ms, bound);
	}

	cudaFree(d_src);
	cudaFree(d_dst);
	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
	return 0;
}
