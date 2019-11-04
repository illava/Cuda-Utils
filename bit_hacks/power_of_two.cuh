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


__device__ __host__ __inline__ unsigned the_next_power_of_2(unsigned v)
{
#ifdef __CUDA_ARCH__
    return 1U << (32 - __clz(v - 1));
#else
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
#endif
}

/// @note different behaviour for v == 0

__device__ __host__ __inline__ unsigned the_previous_power_of_2(unsigned v)
{
#ifdef __CUDA_ARCH__
    return 1U << (31 - __clz(v));
#else
    v /= 2;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
#endif
}
