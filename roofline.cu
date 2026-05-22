/*
 * roofline.cu - measure peak FP32 FLOPS and memory bandwidth
 * RTX 2060 Mobile: ~8 TFLOPS FP32, ~263 GB/s theoretical
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
	cudaError_t e = (x); \
	if (e != cudaSuccess) { \
		fprintf(stderr, "CUDA error %s:%d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(1); \
	} \
} while (0)

/* ------------------------------------------------------------------ */
/* Peak FP32 FLOPS                                                     */
/* 8 independent accumulators per thread to hide FMA latency (~4 cy)  */
/* ------------------------------------------------------------------ */
__global__ void kernel_flops(float *out, long iters)
{
	float a0 = 1.0f, a1 = 1.1f, a2 = 1.2f, a3 = 1.3f;
	float a4 = 1.4f, a5 = 1.5f, a6 = 1.6f, a7 = 1.7f;
	float b  = 1.000001f;

	for (long i = 0; i < iters; i++) {
		a0 = a0 * b + b;
		a1 = a1 * b + b;
		a2 = a2 * b + b;
		a3 = a3 * b + b;
		a4 = a4 * b + b;
		a5 = a5 * b + b;
		a6 = a6 * b + b;
		a7 = a7 * b + b;
	}

	/* prevent dead-code elimination */
	if (out && threadIdx.x == 0 && blockIdx.x == 0)
		*out = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

/* ------------------------------------------------------------------ */
/* Peak memory bandwidth — streaming copy (read + write)              */
/* Both src load and dst store are live; compiler can't eliminate.    */
/* Reported BW = bytes_read + bytes_written = 2 * array_size.        */
/* ------------------------------------------------------------------ */
__global__ void kernel_bw(const float4 * __restrict__ src,
			   float4 * __restrict__ dst,
			   long n)
{
	long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
	long stride = (long)gridDim.x * blockDim.x;

	for (long i = tid; i < n; i += stride)
		dst[i] = src[i];
}

/* ------------------------------------------------------------------ */

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
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

	printf("Device: %s\n", prop.name);
	printf("SMs: %d  maxClockRate: %.0f MHz\n",
		prop.multiProcessorCount, sm_clock_khz / 1e3);
	printf("Mem bus: %d-bit  memClockRate: %.0f MHz\n\n",
		mem_bus_width, mem_clock_khz / 1e3);

	cudaEvent_t t0, t1;
	CUDA_CHECK(cudaEventCreate(&t0));
	CUDA_CHECK(cudaEventCreate(&t1));

	/* ---- FLOPS benchmark ---- */
	{
		/* fill all warps: 15 SM × 2048 resident threads/SM = 30720 */
		int threads = 256;
		int blocks  = prop.multiProcessorCount * 8; /* 8 blocks/SM */
		long iters  = 1 << 14; /* 16384 iterations */

		float *d_out;
		CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

		/* warmup */
		kernel_flops<<<blocks, threads>>>(d_out, iters);
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(t0));
		kernel_flops<<<blocks, threads>>>(d_out, iters);
		CUDA_CHECK(cudaEventRecord(t1));
		CUDA_CHECK(cudaDeviceSynchronize());

		double ms = elapsed_ms(t0, t1);
		/* 8 FMAs/thread/iter = 16 FLOPs/thread/iter */
		long total_threads = (long)blocks * threads;
		double flops = (double)total_threads * iters * 16.0;
		printf("FLOPS benchmark\n");
		printf("  blocks=%d  threads=%d  iters=%ld\n",
			blocks, threads, iters);
		printf("  time:  %.2f ms\n", ms);
		printf("  perf:  %.2f TFLOPS\n\n", flops / ms / 1e9);

		cudaFree(d_out);
	}

	/* ---- Bandwidth benchmark ---- */
	{
		/* 512 MB of float4 data */
		long n = (512L << 20) / sizeof(float4);
		size_t bytes = n * sizeof(float4);

		float4 *d_src, *d_dst;
		CUDA_CHECK(cudaMalloc(&d_src, bytes));
		CUDA_CHECK(cudaMalloc(&d_dst, bytes));
		CUDA_CHECK(cudaMemset(d_src, 0, bytes));

		int threads = 256;
		/* enough blocks to keep device busy but not thrash */
		int blocks  = prop.multiProcessorCount * 4;

		/* warmup */
		kernel_bw<<<blocks, threads>>>(d_src, d_dst, n);
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaEventRecord(t0));
		kernel_bw<<<blocks, threads>>>(d_src, d_dst, n);
		CUDA_CHECK(cudaEventRecord(t1));
		CUDA_CHECK(cudaDeviceSynchronize());

		double ms = elapsed_ms(t0, t1);
		/* copy = 1 read + 1 write = 2 × array bytes transferred */
		double gb  = 2.0 * (double)bytes / (1 << 30);
		printf("Bandwidth benchmark  (streaming copy, read+write)\n");
		printf("  array: %.0f MB  n=%ld float4s\n",
			(double)bytes / (1 << 20), n);
		printf("  time: %.2f ms\n", ms);
		printf("  bw:   %.1f GB/s\n\n", gb / ms * 1e3);

		/* theoretical peak from device props */
		double peak_bw = 2.0 * mem_clock_khz * 1e3
				 * mem_bus_width / 8.0 / 1e9;
		/* theoretical peak FP32: cores * 2 FLOP/FMA * clock */
		double peak_flops = (double)prop.multiProcessorCount
				    * 64.0 /* FP32 units per SM, Turing */
				    * 2.0  /* FMA = 2 FLOPs */
				    * sm_clock_khz * 1e3 / 1e12; /* TFLOPS */
		double ridge = peak_flops * 1e12 / (peak_bw * 1e9); /* FLOP/B */

		printf("Theoretical peaks\n");
		printf("  FP32:  %.2f TFLOPS\n", peak_flops);
		printf("  BW:    %.1f GB/s\n", peak_bw);
		printf("  Ridge: %.1f FLOP/byte\n", ridge);

		cudaFree(d_src);
		cudaFree(d_dst);
	}

	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
	return 0;
}
