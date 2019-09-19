/*
Copyright [2019] [illava(illava@outlook.com)]
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

__global__ void assign(uint32_t *x, uint32_t n) { x[0] = n; }

const int blockSize = 1024;

// The original code

// d_key is the original key, keeps untouched

// d_temp is a copy of d_key, changes during algorithm

// shift = 2 ^ d

// __global__ void uphill(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp,
//                        int64_t n, int64_t shift)
// {
//     int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n)
//     {
//         if (idx % (2 * shift) == 0)
//         {
//             if (d_temp[idx + 2 * shift - 1] == 0)
//                 d_value[idx + 2 * shift - 1] += d_value[idx + shift - 1];
//             d_temp[idx + 2 * shift - 1] |= d_temp[idx + shift - 1];
//         }
//     }
// }

// My adaption

__global__ void uphill1(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp,
                        int64_t n, int64_t shift)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + shift < n)
    {
        if (idx % (2 * shift) == 0)
        {
            if (d_temp[n - 1 - idx] == 0)
                d_value[n - 1 - idx] += d_value[n - 1 - (idx + shift)];
            d_temp[n - 1 - idx] |= d_temp[n - 1 - (idx + shift)];
        }
    }
}

// __global__ void downhill(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp,
//                          int64_t n, int64_t shift)
// {
//     int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n)
//     {
//         if (idx % (2 * shift) == 0)
//         {
//             uint32_t tmp             = d_value[idx + shift - 1];
//             d_value[idx + shift - 1] = d_value[idx + 2 * shift - 1];
//             if (d_key[idx + shift] == 1)
//                 d_value[idx + 2 * shift - 1] = 0;
//             else if (d_temp[idx + shift - 1] == 1)
//                 d_value[idx + 2 * shift - 1] = tmp;
//             else
//                 d_value[idx + 2 * shift - 1] =
//                     tmp + d_value[idx + 2 * shift - 1];
//             d_temp[idx + shift - 1] = 0;
//         }
//     }
// }

__global__ void downhill1(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp,
                          int64_t n, int64_t shift)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + shift < n)
    {
        if (idx % (2 * shift) == 0)
        {
            uint32_t tmp                   = d_value[n - 1 - (idx + shift)];
            d_value[n - 1 - (idx + shift)] = d_value[n - 1 - idx];
            if (d_key[n - 1 - (idx + shift - 1)] == 1)
                d_value[n - 1 - idx] = 0;
            else if (d_temp[n - 1 - (idx + shift)] == 1)
                d_value[n - 1 - idx] = tmp;
            else
                d_value[n - 1 - idx] = tmp + d_value[n - 1 - idx];
            d_temp[n - 1 - (idx + shift)] = 0;
        }
    }
}

// void segscan(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp, int64_t n)
// {
//     cudaMemcpy(d_temp, d_key, n * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
//     int64_t shift;
//     for (shift = 1; shift < n; shift *= 2)
//         uphill<<<n / blockSize + 1, blockSize>>>(d_value, d_key, d_temp, n,
//                                                  shift);
//     assign<<<1, 1>>>(d_value + n - 1, 0U);
//     for (shift /= 2; shift >= 1; shift /= 2)
//         downhill<<<n / blockSize + 1, blockSize>>>(d_value, d_key, d_temp, n,
//                                                    shift);
// }

void segscan1(uint32_t *d_value, uint8_t *d_key, uint8_t *d_temp, int64_t n)
{
    cudaMemcpy(d_temp, d_key, n * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    int64_t shift;
    for (shift = 1; shift < n; shift *= 2)
        uphill1<<<n / blockSize + 1, blockSize>>>(d_value, d_key, d_temp, n,
                                                  shift);
    assign<<<1, 1>>>(d_value + n - 1, 0U);
    for (shift /= 2; shift >= 1; shift /= 2)
        downhill1<<<n / blockSize + 1, blockSize>>>(d_value, d_key, d_temp, n,
                                                    shift);
}

int test()
{
    const int n = 17; // no extra space needed for n not power of 2.

    uint8_t * h_key   = new uint8_t[n];
    uint8_t * h_temp  = new uint8_t[n];
    uint32_t *h_value = new uint32_t[n];
    uint8_t * d_key;
    uint8_t * d_temp;
    uint32_t *d_value;
    cudaMalloc(&d_key, sizeof(uint8_t) * n);
    cudaMalloc(&d_temp, sizeof(uint8_t) * n);
    cudaMalloc(&d_value, sizeof(uint32_t) * n);

    for (int i = 0; i < n; i++) h_key[i] = 0;         // segments
    for (int i = 0; i < n; i++) h_value[i] = i * 10;  // data

    // assigning segments
    h_key[2]  = 1;
    h_key[5]  = 1;
    h_key[7]  = 1;
    h_key[12] = 1;

    int m = min(n, 100);  // print limit

    for (int i = 0; i < m; i++)
    {
        if (h_key[i]) printf("---\n");
        printf("h_value, %d, %d\n", i, h_value[i]);
    }

    cudaMemcpy(d_key, h_key, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, h_temp, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, sizeof(uint32_t) * n, cudaMemcpyHostToDevice);

    segscan1(d_value, d_key, d_temp, n);

    cudaDeviceSynchronize();
    printf("cuda:Error:%s\n", cudaGetErrorString(cudaPeekAtLastError()));

    // cudaMemcpy(h_key, d_key, sizeof(uint8_t) * n, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_temp, d_temp, sizeof(uint8_t) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_value, d_value, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < m; i++) printf("h_key, %d, %d\n", i, h_key[i]);
    for (int i = 0; i < m; i++) printf("h_value, %d, %d\n", i, h_value[i]);
    // for (int i = 0; i < m; i++) printf("h_temp, %d, %d\n", i, h_temp[i]);

    delete[] h_key;
    delete[] h_value;
    delete[] h_temp;

    cudaFree(d_key);
    cudaFree(d_temp);
    cudaFree(d_value);
    return 0;
}

int main()
{
    test();
    return 0;
}