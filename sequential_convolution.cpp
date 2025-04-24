//
// Created by Christian Zanzi on 09/04/25.
//
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <string>

#define FILTER_WIDTH 3
#define FILTER_RADIUS (FILTER_WIDTH / 2)
#define clamp(x) (fmin(fmax((x), 0.0f), 1.0f))

void convolution(float *input, float *output, float *filter, int channels, int width, int height) {
    // image matrix memorized in row-major order, linear memory access to consecutive addresses
    for (int k = 0; k < channels; ++k) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0;

                for (int fy = - FILTER_RADIUS; fy <=  FILTER_RADIUS; ++fy) {
                    for (int fx = - FILTER_RADIUS; fx <= FILTER_RADIUS; ++fx) {
                        // setting the index position of the image tile with respect to the filter
                        int nx = x + fx;
                        int ny = y + fy;

                        // border handling with replicate padding for the ghost cells
                        if (nx < 0) nx = 0;
                        if (ny < 0) ny = 0;
                        if (nx >= width) nx = width - 1;
                        if (ny >= height) ny = height -1;

                        const int idx = (ny * width + nx) * channels + k;
                        sum += input[idx] * filter[(fx + FILTER_RADIUS) * FILTER_WIDTH + (fy + FILTER_RADIUS)];
                    }
                }
                int out_idx = (y * width + x) * channels + k;
                output[out_idx] = sum;
            }
        }
    }
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

    size_t image_size = width * height * channels;
    float *host_input = new float[image_size];
    float *host_output = new float[image_size];
    for (size_t i = 0; i < image_size; ++i)
        host_input[i] = input_image[i] / 255.0f;

    float host_filter[FILTER_WIDTH * FILTER_WIDTH] = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
     };

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    convolution(host_input, host_output, host_filter, channels, width, height);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> milliseconds = end - start;
    std::cout << "Total convolution time: " << milliseconds.count() << " ms" << std::endl;


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
