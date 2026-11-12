

#include <openslide.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <bits/stdc++.h>
using namespace std;

namespace fs = std::filesystem;

//Grayscale Conversion
vector<uint8_t> toGrayscale(const std::vector<uint32_t>& bgra,
                                  int64_t width, int64_t height) {
    vector<uint8_t> gray(width * height);
    for (int64_t i = 0; i < width * height; i++) {
        uint32_t px = bgra[i];
        uint8_t b = (px >> 0)  & 0xFF;
        uint8_t g = (px >> 8)  & 0xFF;
        uint8_t r = (px >> 16) & 0xFF;
        // standard luminance weights
        gray[i] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return gray;
}


//Tile metadata + extraction from the inmemory grayscale buffer

struct TileMeta {
    int tile_row, tile_col;
    int64_t x_offset, y_offset; // top-left, in the grayscale level image
    int width, height;          // actual tile size (boundary tiles may be smaller)
};

vector<TileMeta> getTileGrid(int64_t levelWidth, int64_t levelHeight, int tileSize) {
    vector<TileMeta> tiles;
    int64_t nCols = (levelWidth  + tileSize - 1) / tileSize;
    int64_t nRows = (levelHeight + tileSize - 1) / tileSize;
    for (int64_t row = 0; row < nRows; row++) {
        for (int64_t col = 0; col < nCols; col++) {
            TileMeta t;
            t.tile_row = static_cast<int>(row);
            t.tile_col = static_cast<int>(col);
            t.x_offset = col * tileSize;
            t.y_offset = row * tileSize;
            t.width  = static_cast<int>(std::min<int64_t>(tileSize, levelWidth  - t.x_offset));
            t.height = static_cast<int>(std::min<int64_t>(tileSize, levelHeight - t.y_offset));
            tiles.push_back(t);
        }
    }
    return tiles;
}

std::vector<uint8_t> extractTile(const std::vector<uint8_t>& fullGray,
                                  int64_t fullWidth,
                                  const TileMeta& meta) {
    vector<uint8_t> tile(meta.width * meta.height);
    for (int y = 0; y < meta.height; y++) {
        for (int x = 0; x < meta.width; x++) {
            int64_t srcX = meta.x_offset + x;
            int64_t srcY = meta.y_offset + y;
            tile[y * meta.width + x] = fullGray[srcY * fullWidth + srcX];
        }
    }
    return tile;
}

// Skip tiles that are almost entirely background
// H&E background is near white, so mean intensity will be high
bool tileHasTissue(const std::vector<uint8_t>& tile, uint8_t whiteThreshold = 220) {
    if (tile.empty()) return false;
    double sum = 0;
    for (auto v : tile) sum += v;
    double mean = sum / tile.size();
    return mean < whiteThreshold; // darker average => likely contains tissue
}

// Computing LBP
vector<uint8_t> computeLBP(const std::vector<uint8_t>& gray, int width, int height) {
    vector<uint8_t> lbp(width * height, 0);

    // 8 neighbors and we are clockwise starting from top-left
    static const int dx[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    static const int dy[8] = {-1, -1, -1, 0, 1, 1, 1, 0};

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            uint8_t center = gray[y * width + x];
            uint8_t code = 0;
            for (int p = 0; p < 8; p++) {
                uint8_t neighbor = gray[(y + dy[p]) * width + (x + dx[p])];
                if (neighbor >= center) {
                    code |= (1 << p);
                }
            }
            lbp[y * width + x] = code;
        }
    }
    return lbp;
}

array<int, 256> computeHistogram(const vector<uint8_t>& lbp) {
    array<int, 256> hist{};
    for (auto v : lbp) hist[v]++;
    return hist;
}


void saveGrayPNG(const string& path, const vector<uint8_t>& img, int w, int h) {
    stbi_write_png(path.c_str(), w, h, 1, img.data(), w);
}

void saveHistogramCSV(const string& path, const array<int, 256>& hist) {
    ofstream f(path);
    f << "bin,count\n";
    for (int i = 0; i < 256; i++) f << i << "," << hist[i] << "\n";
}


int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " path/to/slide.svs [tileSize] [maxTilesToProcess]\n";
        return 1;
    }
    string svsPath = argv[1];
    int tileSize = (argc >= 3) ? stoi(argv[2]) : 128;
    int maxTiles = (argc >= 4) ? stoi(argv[3]) : 8;

    fs::create_directories("output");

    //Opening the slide
    openslide_t* osr = openslide_open(svsPath.c_str());
    if (!osr) {
        cerr << "Failed to open slide: " << svsPath << "\n";
        return 1;
    }
    const char* err = openslide_get_error(osr);
    if (err) {
        cerr << "OpenSlide error: " << err << "\n";
        return 1;
    }

    int32_t levelCount = openslide_get_level_count(osr);
    cout << "Slide opened. Pyramid levels: " << levelCount << "\n";

    int64_t w0, h0;
    openslide_get_level0_dimensions(osr, &w0, &h0);
    cout << "Level 0 dimensions: " << w0 << " x " << h0 << "\n";

    // Picking the smallest level whose width is still reasonably large
    // We want at least ~1000px so tiling is meaningful, but small
    // enough to comfortably fit in memory.
    int chosenLevel = levelCount - 1; // start from the smallest
    int64_t chosenW = 0, chosenH = 0;
    for (int32_t lvl = 0; lvl < levelCount; lvl++) {
        int64_t lw, lh;
        openslide_get_level_dimensions(osr, lvl, &lw, &lh);
        cout << "  Level " << lvl << ": " << lw << " x " << lh << "\n";
        if (lw <= 4000 && lw > chosenW) { // pick the largest level under ~4000px wide
            chosenLevel = lvl;
            chosenW = lw;
            chosenH = lh;
        }
    }
    if (chosenW == 0) { // fallback: just use the smallest level available
        openslide_get_level_dimensions(osr, levelCount - 1, &chosenW, &chosenH);
        chosenLevel = levelCount - 1;
    }
    cout << "Chosen level for demo: " << chosenLevel
              << " (" << chosenW << " x " << chosenH << ")\n";

    //Reading that whole level into memory
    vector<uint32_t> buffer(chosenW * chosenH);
    openslide_read_region(osr, buffer.data(), 0, 0, chosenLevel, chosenW, chosenH);
    err = openslide_get_error(osr);
    if (err) {
        cerr << "OpenSlide read_region error: " << err << "\n";
        return 1;
    }

    //Grayscale conversion
    vector<uint8_t> gray = toGrayscale(buffer, chosenW, chosenH);
    saveGrayPNG("output/level_grayscale_full.png", gray, chosenW, chosenH);
    cout << "Saved full-level grayscale image to output/level_grayscale_full.png\n";

    //Tiling
    vector<TileMeta> tiles = getTileGrid(chosenW, chosenH, tileSize);
    cout << "Tile grid: " << tiles.size() << " tiles of size "
              << tileSize << "x" << tileSize << "\n";

    //LBP per tile
    int processed = 0;
    for (const auto& meta : tiles) {
        if (processed >= maxTiles) break;

        std::vector<uint8_t> tile = extractTile(gray, chosenW, meta);
        if (!tileHasTissue(tile)) continue; // skip background-only tiles

        vector<uint8_t> lbp = computeLBP(tile, meta.width, meta.height);
        array<int, 256> hist = computeHistogram(lbp);

        string base = "output/tile_r" + to_string(meta.tile_row) +
                            "_c" + to_string(meta.tile_col);
        saveGrayPNG(base + "_original.png", tile, meta.width, meta.height);
        saveGrayPNG(base + "_lbp.png", lbp, meta.width, meta.height);
        saveHistogramCSV(base + "_histogram.csv", hist);

        cout << "Processed tile (row=" << meta.tile_row
                  << ", col=" << meta.tile_col << ") -> " << base << "_*.png/csv\n";
        processed++;
    }

    cout << "\nDone. Processed " << processed << " tissue-containing tiles.\n";
    cout << "Check the 'output/' folder for:\n";
    cout << "  - level_grayscale_full.png   (whole chosen level, grayscale)\n";
    cout << "  - tile_rX_cY_original.png    (original tile)\n";
    cout << "  - tile_rX_cY_lbp.png         (LBP output image)\n";
    cout << "  - tile_rX_cY_histogram.csv   (256-bin LBP histogram)\n";

    openslide_close(osr);
    return 0;
}
