#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define FILTER_WIDTH 3
#define FILTER_RADIUS (FILTER_WIDTH / 2)
#define TILE_WIDTH 32
#define w (TILE_WIDTH + FILTER_WIDTH - 1)
#define clamp(x) (min(max((x), 0.0f), 1.0f))

__constant__ float device_filter[FILTER_WIDTH * FILTER_WIDTH];

__global__ void convolution(float *input, float *output,
                            int channels, int width, int height) {
   __shared__ float N_s[w][w];

   for (int k = 0; k < channels; k++) {
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
          destY = dest / w, destX = dest % w,
          srcY = blockIdx.y * TILE_WIDTH + destY - FILTER_RADIUS,
          srcX = blockIdx.x * TILE_WIDTH + destX - FILTER_RADIUS,
          src = (srcY * width + srcX) * channels + k;

      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_s[destY][destX] = input[src];
      else
         N_s[destY][destX] = 0.0f;

      dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
      destY = dest / w; destX = dest % w;
      srcY = blockIdx.y * TILE_WIDTH + destY - FILTER_RADIUS;
      srcX = blockIdx.x * TILE_WIDTH + destX - FILTER_RADIUS;
      src = (srcY * width + srcX) * channels + k;

      if (destY < w) {
         if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_s[destY][destX] = input[src];
         else
            N_s[destY][destX] = 0.0f;
      }
      __syncthreads();

      float accum = 0.0f;
      for (int fy = 0; fy < FILTER_WIDTH; fy++)
         for (int fx = 0; fx < FILTER_WIDTH; fx++)
            accum += N_s[threadIdx.y + fy][threadIdx.x + fx] * device_filter[fy * FILTER_WIDTH + fx];

      int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         output[(y * width + x) * channels + k] = clamp(accum);
      __syncthreads();
   }
}

void launch_convolution(float *host_input, float *host_filter, float *host_output, int channels, int width, int height) {
   float *device_input, *device_output;
   size_t image_size = width * height * channels * sizeof(float);
   size_t filter_size = FILTER_WIDTH * FILTER_WIDTH * sizeof(float);

   cudaMalloc(&device_input, image_size);
   cudaMalloc(&device_output, image_size);

   cudaMemcpy(device_input, host_input, image_size, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(device_filter, host_filter, filter_size);

   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
   dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

   std::chrono::time_point<std::chrono::system_clock> start, end;
   start = std::chrono::system_clock::now();
   convolution<<<dimGrid, dimBlock>>>(device_input, device_output, channels, width, height);

   cudaDeviceSynchronize();
   end = std::chrono::system_clock::now();
   std::chrono::duration<double> milliseconds = end - start;
   std::cout << "Total convolution time: " << milliseconds.count() << " ms" << std::endl;

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
   }

   cudaMemcpy(host_output, device_output, image_size, cudaMemcpyDeviceToHost);


   cudaFree(device_input);
   cudaFree(device_output);
}

int main(int argc, char** argv) {
   if (argc < 3) {
      std::cerr << "Usage: ./convolve image.jpg output.jpg\n";
      return 1;
   }

   int width, height, channels;
   unsigned char *input_image = stbi_load(argv[1], &width, &height, &channels, 0);
   if (!input_image) {
      std::cerr << "Failed to load image\n";
      return 1;
   }

   // Convert input to float
   size_t image_size = width * height * channels;
   float *host_input = new float[image_size];
   float *host_output = new float[image_size];
   for (size_t i = 0; i < image_size; ++i)
      host_input[i] = input_image[i] / 255.0f;

   // Edge Detection filter
   float host_filter[FILTER_WIDTH * FILTER_WIDTH] = {
      -1, -1, -1,
      -1, 8, -1,
      -1, -1, -1
   };

   launch_convolution(host_input, host_filter, host_output, channels, width, height);

   // Convert back to unsigned char
   unsigned char *output_image = new unsigned char[image_size];
   for (size_t i = 0; i < image_size; ++i)
      output_image[i] = static_cast<unsigned char>(clamp(host_output[i]) * 255.0f);

   stbi_write_jpg(argv[2], width, height, channels, output_image, 100);

   stbi_image_free(input_image);
   delete[] host_input;
   delete[] host_output;
   delete[] output_image;

   std::cout << "Image saved to " << argv[2] << std::endl;
   return 0;
}