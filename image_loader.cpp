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


struct STBImage {
    int width{0}, height{0}, channels{0};
    uint8_t *rgb_image{nullptr};
    std::string filename;

    STBImage loadImage(const std::string &filename) {
        STBImage img;
        size_t slash = filename.find_last_of("/\\");
        img.rgb_image = stbi_load(filename.c_str(), &img.width, &img.height, &img.channels, 0);
        if (!img.rgb_image) {
            std::cerr << "Errore nel caricamento immagine: " << filename << "\n";
            exit(1);
        }
        img.filename = filename.substr(slash + 1);
        std::cout << "Caricata " << img.filename << " (" << img.width << "x" << img.height << ", " << img.channels << " canali)\n";
        return img;
    }

    void saveImage(const std::string& newName) const {
        int success = stbi_write_jpg(newName.c_str(), width, height, channels, rgb_image, width * channels);
        if (!success) {
            std::cerr << "Errore: impossibile salvare l'immagine in '" << newName << "'\n";
        } else {
            std::cout << "Immagine salvata in '" << newName << "'\n";
        }
    }
};

STBImage convolution(const STBImage& input, const std::vector<float>& filter, int size) {
    STBImage output;
    output.filename = input.filename;
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.rgb_image = new uint8_t[input.width * input.height * input.channels];

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int c = 0; c < input.channels; ++c) {
                float sum = 0;

                for (int fy = - size / 2; fy <=  size / 2; ++fy) {
                    for (int fx = - size / 2; fx <= size / 2; ++fx) {
                        int nx = x + fx;
                        int ny = y + fy;

                        if (nx < 0) nx = 0;
                        if (ny < 0) ny = 0;
                        if (nx >= input.width) nx = input.width - 1;
                        if (ny >= input.height) ny = input.height - 1;

                        int idx = (ny * input.width + nx) * input.channels + c;
                        sum += input.rgb_image[idx] * filter[(fx + size / 2) * size + (fy + size / 2)];
                    }
                }

                int result = static_cast<int>(sum);
                if (result < 0) result = 0;
                if (result > 255) result = 255;

                int out_idx = (y * input.width + x) * input.channels + c;
                output.rgb_image[out_idx] = static_cast<uint8_t>(result);
            }
        }
    }
    return output;
}


int main() {
    std::vector<std::string> filenames = {
        "conv/img1.jpg",
        "conv/img2.jpg",
        "conv/img3.jpg",
        "conv/img4.jpg"
    };

    std::vector<STBImage> images;

    /*std::vector<float> edge_detection = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    };
    int filter_size = 3;*/
    std::vector<float> gaussian_blur = {
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1
    };
    for (auto& k : gaussian_blur) k /= 256;
    int filter_size = 5;


    for (const auto& filename : filenames) {
        STBImage img;
        images.push_back(img.loadImage(filename));
    }

    for (auto& img : images) {
        //img = convolution(img, eedge_detection, filter_size);
        img = convolution(img, gaussian_blur, filter_size);
        img.saveImage(img.filename);
        stbi_image_free(img.rgb_image);
    }

    return 0;
}
