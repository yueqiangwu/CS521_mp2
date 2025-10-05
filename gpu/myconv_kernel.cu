#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8
#define TILE_W 8
#define TILE_C 16

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float *__restrict__ x, // input: N x C x H x W
    const float *__restrict__ w, // weights: C_out x C_in x KH x KW
    float *__restrict__ out,     // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w)
{
  extern __shared__ float shmem[]; // shared memory for partial sums

  // TO DO : Tiled matrix multiplication by using shmem

  const int tile_x = blockIdx.x * TILE_W;
  const int tile_y = blockIdx.y * TILE_H;
  const int n = blockIdx.z;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int thread_linear = ty * blockDim.x + tx;
  const int numThreads = blockDim.x * blockDim.y;

  const int out_x = tile_x + tx;
  const int out_y = tile_y + ty;

  const int in_tile_h = (TILE_H - 1) * stride + KH;
  const int in_tile_w = (TILE_W - 1) * stride + KW;

  const int sh_x_elems = TILE_C * in_tile_h * in_tile_w;

  float *sh_x = shmem;
  float *sh_w = shmem + sh_x_elems;

  bool is_out_valid = (out_x < out_w) && (out_y < out_h);

  for (int co = 0; co < C_out; ++co)
  {
    float acc = 0.0f;

    for (int ci_tile = 0; ci_tile < C_in; ci_tile += TILE_C)
    {
      const int curC = min(TILE_C, C_in - ci_tile);

      int total_x_load = curC * in_tile_h * in_tile_w;
      for (int idx = thread_linear; idx < total_x_load; idx += numThreads)
      {
        int tmp = idx;
        int c_local = tmp / (in_tile_h * in_tile_w);
        tmp %= (in_tile_h * in_tile_w);
        int iy = tmp / in_tile_w;
        int ix = tmp % in_tile_w;

        int in_y_global = tile_y * stride - pad + iy;
        int in_x_global = tile_x * stride - pad + ix;

        float val = 0.0f;
        if ((in_y_global >= 0) && (in_y_global < H) && (in_x_global >= 0) && (in_x_global < W))
        {
          int ci = ci_tile + c_local;
          val = x[((n * C_in + ci) * H + in_y_global) * W + in_x_global];
        }
        int sh_x_idx = (c_local * in_tile_h + iy) * in_tile_w + ix;
        sh_x[sh_x_idx] = val;
      }

      int total_w_load = curC * KH * KW;
      for (int idx = thread_linear; idx < total_w_load; idx += numThreads)
      {
        int tmp = idx;
        int c_local = tmp / (KH * KW);
        tmp %= (KH * KW);
        int kh = tmp / KW;
        int kw = tmp % KW;
        int ci = ci_tile + c_local;
        float wval = w[((co * C_in + ci) * KH + kh) * KW + kw];
        int sh_w_idx = (c_local * KH + kh) * KW + kw;
        sh_w[sh_w_idx] = wval;
      }

      __syncthreads();

      if (is_out_valid)
      {
        for (int c_local = 0; c_local < curC; ++c_local)
        {
          int base_in_c = c_local * in_tile_h * in_tile_w;
          int base_w_c = c_local * KH * KW;

          for (int kh = 0; kh < KH; ++kh)
          {
            int iy = ty * stride + kh;

            int row_offset = iy * in_tile_w;
            for (int kw = 0; kw < KW; ++kw)
            {
              int ix = tx * stride + kw;
              float xval = sh_x[base_in_c + row_offset + ix];
              float wval = sh_w[base_w_c + kh * KW + kw];
              acc += xval * wval;
            }
          }
        }
      }

      __syncthreads();
    }

    if (is_out_valid)
    {
      out[((n * C_out + co) * out_h + out_y) * out_w + out_x] = acc;
    }
  }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                        int stride, int pad)
{
  int N = x.size(0);
  int C_in = x.size(1);
  int H = x.size(2);
  int W = x.size(3);

  int C_out = w.size(0);
  int KH = w.size(2);
  int KW = w.size(3);

  // int out_h = ...
  // int out_w = ...
  int out_h = (H + 2 * pad - KH) / stride + 1;
  int out_w = (W + 2 * pad - KW) / stride + 1;

  auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

  dim3 block(TILE_W, TILE_H);
  dim3 grid((out_w + TILE_W - 1) / TILE_W,
            (out_h + TILE_H - 1) / TILE_H,
            N);

  int patch_h = (TILE_H - 1) * stride + KH;
  int patch_w = (TILE_W - 1) * stride + KW;

  size_t patch_size = (size_t)TILE_C * patch_h * patch_w;
  size_t w_size = (size_t)TILE_C * KH * KW;
  size_t shmem_size = (patch_size + w_size) * sizeof(float);

  gemm_gpu_o4_kernel<<<grid, block, shmem_size>>>(
      x.data_ptr<float>(),
      w.data_ptr<float>(),
      out.data_ptr<float>(),
      N, C_in, H, W,
      C_out, KH, KW,
      stride, pad,
      out_h, out_w);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
