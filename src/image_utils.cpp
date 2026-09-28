#include "image_utils.h"

#include <fstream>
#include <sstream>

namespace {

bool ReadToken(std::istream& input, std::string* token) {
    while (input >> *token) {
        if (!token->empty() && (*token)[0] == '#') {
            std::string ignored;
            std::getline(input, ignored);
            continue;
        }
        return true;
    }
    return false;
}

}  // namespace

bool ReadPgm(const std::string& path, PgmImage* image, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        *error = "Unable to open " + path;
        return false;
    }

    std::string magic;
    std::string width_text;
    std::string height_text;
    std::string max_value_text;

    if (!ReadToken(input, &magic) ||
        !ReadToken(input, &width_text) ||
        !ReadToken(input, &height_text) ||
        !ReadToken(input, &max_value_text)) {
        *error = "Invalid PGM header in " + path;
        return false;
    }

    if (magic != "P5") {
        *error = "Only binary PGM (P5) images are supported: " + path;
        return false;
    }

    int width = 0;
    int height = 0;
    int max_value = 0;

    try {
        width = std::stoi(width_text);
        height = std::stoi(height_text);
        max_value = std::stoi(max_value_text);
    } catch (...) {
        *error = "Invalid numeric PGM header in " + path;
        return false;
    }

    if (width <= 0 || height <= 0 || max_value != 255) {
        *error = "PGM must have positive dimensions and max value 255: " + path;
        return false;
    }

    input.get();  // Consume the single whitespace byte after max value.

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) * height;
    std::vector<unsigned char> pixels(pixel_count);
    input.read(reinterpret_cast<char*>(pixels.data()),
               static_cast<std::streamsize>(pixel_count));

    if (input.gcount() != static_cast<std::streamsize>(pixel_count)) {
        *error = "PGM pixel data is incomplete: " + path;
        return false;
    }

    image->width = width;
    image->height = height;
    image->pixels = std::move(pixels);
    return true;
}

bool WritePgm(const std::string& path,
              int width,
              int height,
              const unsigned char* pixels,
              std::string* error) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        *error = "Unable to create " + path;
        return false;
    }

    output << "P5\n" << width << ' ' << height << "\n255\n";
    output.write(reinterpret_cast<const char*>(pixels),
                 static_cast<std::streamsize>(
                     static_cast<std::size_t>(width) * height));

    if (!output) {
        *error = "Failed while writing " + path;
        return false;
    }

    return true;
}
