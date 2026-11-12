// main_gabor.cpp
// ---------------------------------------------------------------------
// SVS -> grayscale -> tiles (every pyramid level) -> raw Gabor filters
//     -> response pictures + feature numbers
//
// Build:
//   g++ -std=c++17 -O2 main_gabor.cpp $(pkg-config --cflags --libs openslide) -o gabor_demo
//
// Run on a slide:
//   ./gabor_demo slide.svs [tileSize] [maxTilesPerLevel] [whiteThreshold]
//   example: ./gabor_demo slide.svs 256 6 220
//
// Run the built-in proof that the filter works (no slide needed):
//   ./gabor_demo --selftest
//
// Files used:
//   gabor.h            <- the raw Gabor code (the heart of the project)
//   stb_image_write.h  <- tiny library, only used to save PNG pictures
//   OpenSlide          <- library, only used to read the .svs file
// ---------------------------------------------------------------------

#include <openslide.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "gabor.h"

#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// =====================================================================
// PART 1: the filter bank (8 filters = 2 stripe widths x 4 directions)
// =====================================================================
struct Filter {
    std::string name;   // e.g. "L8_T45" = wavelength 8, direction 45 degrees
    GaborParams params;
    Kernel kernel;
};

std::vector<Filter> buildFilterBank() {
    const double lambdas[2] = {4.0, 8.0};               // thin stripes, thick stripes
    const int    angles[4]  = {0, 45, 90, 135};         // directions in degrees
    std::vector<Filter> bank;
    for (double lambda : lambdas) {
        for (int deg : angles) {
            Filter f;
            f.params.lambda = lambda;
            f.params.theta  = deg * kPi / 180.0;        // degrees -> radians
            f.params.sigma  = 0.56 * lambda;            // common rule: bubble ~ 2 stripes wide
            f.params.gamma  = 0.5;                      // bubble stretched along the stripes
            f.params.psi    = 0.0;                      // no phase shift
            f.name = "L" + std::to_string(static_cast<int>(lambda)) +
                     "_T" + std::to_string(deg);
            f.kernel = makeGaborKernel(f.params);       // built ONCE, reused for every tile
            bank.push_back(f);
        }
    }
    return bank;
}

// Biggest "half width" among all kernels. Statistics ignore this much
// border on every side of the tile.
int maxMargin(const std::vector<Filter>& bank) {
    int m = 0;
    for (const auto& f : bank) m = std::max(m, f.kernel.size / 2);
    return m;
}

// =====================================================================
// PART 2: grayscale + reading tiles directly from OpenSlide
// =====================================================================
// OpenSlide gives each pixel as one 32-bit number laid out as A R G B
// (on a normal PC the bytes in memory are B, G, R, A).
std::vector<uint8_t> toGrayscale(const std::vector<uint32_t>& bgra, int64_t n) {
    std::vector<uint8_t> gray(n);
    for (int64_t i = 0; i < n; i++) {
        uint32_t px = bgra[i];
        uint8_t b = (px >> 0)  & 0xFF;
        uint8_t g = (px >> 8)  & 0xFF;
        uint8_t r = (px >> 16) & 0xFF;
        gray[i] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return gray;
}

struct TileMeta {
    int tile_row, tile_col;
    int64_t x_offset, y_offset;   // top-left corner, in THIS level's pixels
    int width, height;            // edge tiles can be smaller than tileSize
};

std::vector<TileMeta> getTileGrid(int64_t levelW, int64_t levelH, int tileSize) {
    std::vector<TileMeta> tiles;
    int64_t nCols = (levelW + tileSize - 1) / tileSize;   // ceiling division
    int64_t nRows = (levelH + tileSize - 1) / tileSize;
    for (int64_t r = 0; r < nRows; r++) {
        for (int64_t c = 0; c < nCols; c++) {
            TileMeta t;
            t.tile_row = (int)r;
            t.tile_col = (int)c;
            t.x_offset = c * tileSize;
            t.y_offset = r * tileSize;
            t.width    = (int)std::min<int64_t>(tileSize, levelW - t.x_offset);
            t.height   = (int)std::min<int64_t>(tileSize, levelH - t.y_offset);
            tiles.push_back(t);
        }
    }
    return tiles;
}

// Read ONE tile straight from the file (never the whole level), so this is
// safe even for level 0 of a 100,000 x 100,000 slide.
// OpenSlide wants x,y in LEVEL-0 coordinates, so we multiply by the
// level's downsample factor. width/height stay in this level's pixels.
std::vector<uint8_t> readTileGray(openslide_t* osr, int32_t level, const TileMeta& t) {
    double ds = openslide_get_level_downsample(osr, level);
    int64_t x0 = (int64_t)std::llround(t.x_offset * ds);
    int64_t y0 = (int64_t)std::llround(t.y_offset * ds);
    std::vector<uint32_t> buf((size_t)t.width * t.height);
    openslide_read_region(osr, buf.data(), x0, y0, level, t.width, t.height);
    return toGrayscale(buf, (int64_t)t.width * t.height);
}

// Fraction of pixels that are darker than whiteThreshold (= "look like tissue").
double tissueFraction(const std::vector<uint8_t>& g, uint8_t whiteThreshold) {
    if (g.empty()) return 0;
    size_t count = 0;
    for (auto v : g) if (v < whiteThreshold) count++;
    return (double)count / g.size();
}

// ---- A small low-resolution picture of the whole slide ("thumbnail") ----
// We use it to find where the tissue is WITHOUT reading any big tiles.
struct Thumbnail {
    std::vector<uint8_t> gray;
    int64_t w = 0, h = 0;
    double downsample = 1.0;
};

Thumbnail loadThumbnail(openslide_t* osr) {
    Thumbnail t;
    int32_t level = openslide_get_level_count(osr) - 1;   // the smallest level
    openslide_get_level_dimensions(osr, level, &t.w, &t.h);
    t.downsample = openslide_get_level_downsample(osr, level);
    std::vector<uint32_t> buf((size_t)t.w * t.h);
    openslide_read_region(osr, buf.data(), 0, 0, level, t.w, t.h);
    t.gray = toGrayscale(buf, t.w * t.h);
    return t;
}

// Does the area covered by this tile contain tissue, according to the thumbnail?
bool thumbnailSaysTissue(const Thumbnail& th, double levelDownsample, const TileMeta& t,
                         uint8_t whiteThreshold, double minFraction) {
    // the tile's box in level-0 pixels, then in thumbnail pixels
    double x0 = t.x_offset * levelDownsample / th.downsample;
    double y0 = t.y_offset * levelDownsample / th.downsample;
    double x1 = (t.x_offset + t.width)  * levelDownsample / th.downsample;
    double y1 = (t.y_offset + t.height) * levelDownsample / th.downsample;
    int ix0 = std::max(0, (int)std::floor(x0));
    int iy0 = std::max(0, (int)std::floor(y0));
    int ix1 = std::min((int)th.w, std::max(ix0 + 1, (int)std::ceil(x1)));
    int iy1 = std::min((int)th.h, std::max(iy0 + 1, (int)std::ceil(y1)));
    size_t total = 0, dark = 0;
    for (int y = iy0; y < iy1; y++)
        for (int x = ix0; x < ix1; x++) {
            total++;
            if (th.gray[(size_t)y * th.w + x] < whiteThreshold) dark++;
        }
    return total > 0 && (double)dark / total >= minFraction;
}

// =====================================================================
// PART 3: running the whole filter bank on one tile
// =====================================================================
struct TileResult {
    std::vector<std::vector<double>> responses;  // 8 response images (w*h each)
    std::vector<double> features;                // 8 filters x 3 numbers = 24
};

TileResult runGaborBank(const std::vector<uint8_t>& tile, int w, int h,
                        const std::vector<Filter>& bank, int margin) {
    TileResult res;
    for (const auto& f : bank) {
        std::vector<double> r = convolve2D(tile, w, h, f.kernel);   // the stripe detector
        ResponseStats st = computeStats(r, w, h, margin);           // 3 summary numbers
        res.features.push_back(st.mean);
        res.features.push_back(st.variance);
        res.features.push_back(st.energy);
        res.responses.push_back(std::move(r));
    }
    return res;
}

std::vector<std::string> featureNames(const std::vector<Filter>& bank) {
    std::vector<std::string> names;
    for (const auto& f : bank) {
        names.push_back("mean_" + f.name);
        names.push_back("var_" + f.name);
        names.push_back("energy_" + f.name);
    }
    return names;
}

// =====================================================================
// PART 4: saving pictures (only for showing results)
// =====================================================================
// Turn a response (which can be negative) into a 0..255 picture using |value|.
// All 8 responses of a tile share the same 'maxAbs', so a brighter picture
// really means a stronger response.
std::vector<uint8_t> responseToImage(const std::vector<double>& r, double maxAbs) {
    std::vector<uint8_t> img(r.size());
    for (size_t i = 0; i < r.size(); i++) {
        double v = (maxAbs > 0) ? std::fabs(r[i]) / maxAbs : 0.0;
        img[i] = (uint8_t)std::min(255.0, v * 255.0);
    }
    return img;
}

// Put several same-size pictures into one big grid picture.
std::vector<uint8_t> makeMontage(const std::vector<std::vector<uint8_t>>& cells,
                                 int cellW, int cellH, int cols, int& outW, int& outH) {
    int rows = ((int)cells.size() + cols - 1) / cols;
    outW = cols * cellW;
    outH = rows * cellH;
    std::vector<uint8_t> m((size_t)outW * outH, 0);
    for (size_t i = 0; i < cells.size(); i++) {
        int ox = ((int)i % cols) * cellW;
        int oy = ((int)i / cols) * cellH;
        for (int y = 0; y < cellH; y++)
            for (int x = 0; x < cellW; x++)
                m[(size_t)(oy + y) * outW + (ox + x)] = cells[i][(size_t)y * cellW + x];
    }
    return m;
}

void savePNG(const std::string& path, const std::vector<uint8_t>& img, int w, int h) {
    stbi_write_png(path.c_str(), w, h, 1, img.data(), w);
}

// Montage layout (5 columns x 2 rows), same order as the kernel picture:
//
//   [ original | L4_T0  | L4_T45  | L4_T90  | L4_T135 ]     <- thin stripes  (wavelength 4)
//   [  (blank) | L8_T0  | L8_T45  | L8_T90  | L8_T135 ]     <- thick stripes (wavelength 8)
//
// Columns 2-5 are the directions 0, 45, 90, 135 degrees.
// A BRIGHT cell means: "this tile has stripes in that direction / width".
void saveTileMontage(const std::string& path, const std::vector<uint8_t>& tile,
                     int w, int h, const TileResult& res) {
    double maxAbs = 0;
    for (const auto& r : res.responses)
        for (double v : r) maxAbs = std::max(maxAbs, std::fabs(v));
    std::vector<uint8_t> blank((size_t)w * h, 0);
    std::vector<std::vector<uint8_t>> cells;
    cells.push_back(tile);
    for (int i = 0; i < 4; i++) cells.push_back(responseToImage(res.responses[i], maxAbs));
    cells.push_back(blank);
    for (int i = 4; i < 8; i++) cells.push_back(responseToImage(res.responses[i], maxAbs));
    int mw, mh;
    auto m = makeMontage(cells, w, h, 5, mw, mh);
    savePNG(path, m, mw, mh);
}

// A picture of the 8 kernels themselves (grey = 0, white = +, black = -).
void saveKernelMontage(const std::string& path, const std::vector<Filter>& bank) {
    const int cell = 110;
    std::vector<std::vector<uint8_t>> cells;
    for (const auto& f : bank) {
        double maxAbs = 0;
        for (double v : f.kernel.w) maxAbs = std::max(maxAbs, std::fabs(v));
        std::vector<uint8_t> img((size_t)cell * cell);
        for (int y = 0; y < cell; y++)
            for (int x = 0; x < cell; x++) {
                int kx = x * f.kernel.size / cell;       // nearest-neighbour scaling
                int ky = y * f.kernel.size / cell;
                double v = f.kernel.w[(size_t)ky * f.kernel.size + kx] / maxAbs;
                img[(size_t)y * cell + x] = (uint8_t)(128 + 127 * v);
            }
        cells.push_back(img);
    }
    int mw, mh;
    auto m = makeMontage(cells, cell, cell, 4, mw, mh);
    savePNG(path, m, mw, mh);
}

// =====================================================================
// PART 5: --selftest  (proof the filter really detects direction)
// =====================================================================
// We draw pictures of pure stripes in 4 known directions. A correct Gabor
// bank must respond MOST strongly with the filter that has the same
// direction, and the table below shows whether that happens.
int runSelfTest() {
    std::cout << "SELF-TEST: synthetic stripe images (wavelength 8 px)\n";
    std::vector<Filter> bank = buildFilterBank();
    int margin = maxMargin(bank);
    fs::create_directories("output_gabor_selftest");
    saveKernelMontage("output_gabor_selftest/gabor_kernels.png", bank);

    const int W = 160, H = 160;
    const int stripeAngles[4] = {0, 45, 90, 135};
    int correct = 0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\nmean |response| of each filter (rows = stripe direction in the image)\n";
    std::cout << std::setw(14) << "image \\ filter";
    for (const auto& f : bank) std::cout << std::setw(9) << f.name;
    std::cout << "   strongest\n";

    for (int deg : stripeAngles) {
        double phi = deg * kPi / 180.0;
        std::vector<uint8_t> img((size_t)W * H);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                double d = x * std::cos(phi) + y * std::sin(phi);   // distance along the wave
                img[(size_t)y * W + x] = (uint8_t)(128 + 100 * std::sin(2 * kPi * d / 8.0));
            }
        TileResult res = runGaborBank(img, W, H, bank, margin);

        std::cout << std::setw(11) << ("stripes " + std::to_string(deg)) << "   ";
        int best = 0;
        for (size_t i = 0; i < bank.size(); i++) {
            double meanAbs = res.features[i * 3];               // feature 0 of each filter
            std::cout << std::setw(9) << meanAbs;
            if (meanAbs > res.features[best * 3]) best = (int)i;
        }
        std::cout << "   " << bank[best].name << "\n";
        if (bank[best].params.theta == phi) correct++;

        saveTileMontage("output_gabor_selftest/stripes_" + std::to_string(deg) + ".png",
                        img, W, H, res);
    }
    std::cout << "\nCorrect direction picked for " << correct << " / 4 stripe images.\n";
    std::cout << "Pictures saved in output_gabor_selftest/\n";
    return correct == 4 ? 0 : 1;
}

// =====================================================================
// MAIN
// =====================================================================
int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") return runSelfTest();

    if (argc < 2) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " slide.svs [tileSize=256] [maxTilesPerLevel=6] [whiteThreshold=220]\n  "
                  << argv[0] << " --selftest\n";
        return 1;
    }
    std::string svsPath = argv[1];
    int tileSize         = (argc >= 3) ? std::stoi(argv[2]) : 256;
    int maxTilesPerLevel = (argc >= 4) ? std::stoi(argv[3]) : 6;
    uint8_t whiteThr     = (argc >= 5) ? (uint8_t)std::stoi(argv[4]) : 220;

    // ---------- open the slide ----------
    openslide_t* osr = openslide_open(svsPath.c_str());
    if (!osr || openslide_get_error(osr)) {
        std::cerr << "Could not open slide: " << svsPath << "\n";
        return 1;
    }
    int32_t levelCount = openslide_get_level_count(osr);
    std::cout << "Slide opened. Pyramid levels: " << levelCount << "\n";

    // ---------- build the 8 Gabor filters once ----------
    std::vector<Filter> bank = buildFilterBank();
    int margin = maxMargin(bank);
    std::vector<std::string> fnames = featureNames(bank);
    std::cout << "Filter bank: " << bank.size() << " filters, biggest kernel "
              << (margin * 2 + 1) << "x" << (margin * 2 + 1) << " pixels\n";

    fs::create_directories("output_gabor");
    saveKernelMontage("output_gabor/gabor_kernels.png", bank);

    Thumbnail thumb = loadThumbnail(osr);

    // CSV with one row per tile, one with one row per level
    std::ofstream tileCsv("output_gabor/gabor_tile_features.csv");
    tileCsv << "level,tile_row,tile_col";
    for (auto& n : fnames) tileCsv << "," << n;
    tileCsv << "\n";

    std::ofstream levelCsv("output_gabor/gabor_level_features.csv");
    levelCsv << "level,tiles_used";
    for (auto& n : fnames) levelCsv << "," << n;
    levelCsv << "\n";

    // ---------- go through EVERY pyramid level ----------
    for (int32_t level = 0; level < levelCount; level++) {
        int64_t lw, lh;
        openslide_get_level_dimensions(osr, level, &lw, &lh);
        double ds = openslide_get_level_downsample(osr, level);
        std::cout << "\n=== Level " << level << " (" << lw << " x " << lh
                  << ", downsample " << ds << "x) ===\n";

        // 1) the full tile grid for this level
        std::vector<TileMeta> grid = getTileGrid(lw, lh, tileSize);

        // 2) keep only tiles that the thumbnail says contain tissue (cheap, no disk reads)
        std::vector<TileMeta> candidates;
        for (const auto& t : grid) {
            if (t.width <= 2 * margin || t.height <= 2 * margin) continue;   // too small to filter
            if (thumbnailSaysTissue(thumb, ds, t, whiteThr, 0.5)) candidates.push_back(t);
        }
        std::cout << "  Tiles in grid: " << grid.size()
                  << ", with tissue: " << candidates.size() << "\n";

        // 3) pick maxTilesPerLevel tiles spread evenly over the tissue
        std::vector<TileMeta> chosen;
        size_t want = std::min<size_t>(maxTilesPerLevel, candidates.size());
        for (size_t i = 0; i < want; i++) {
            size_t idx = (want == 1) ? 0 : i * (candidates.size() - 1) / (want - 1);
            chosen.push_back(candidates[idx]);
        }

        // 4) run the Gabor bank on each chosen tile
        std::vector<double> levelSum(fnames.size(), 0.0);
        int used = 0;
        for (const auto& t : chosen) {
            std::vector<uint8_t> tile = readTileGray(osr, level, t);
            if (tissueFraction(tile, whiteThr) < 0.5) continue;   // double-check the real pixels

            TileResult res = runGaborBank(tile, t.width, t.height, bank, margin);

            std::string base = "output_gabor/level" + std::to_string(level) +
                               "_r" + std::to_string(t.tile_row) +
                               "_c" + std::to_string(t.tile_col);
            saveTileMontage(base + "_gabor.png", tile, t.width, t.height, res);

            tileCsv << level << "," << t.tile_row << "," << t.tile_col;
            for (size_t i = 0; i < res.features.size(); i++) {
                tileCsv << "," << res.features[i];
                levelSum[i] += res.features[i];
            }
            tileCsv << "\n";
            used++;
            std::cout << "  Tile (row=" << t.tile_row << ", col=" << t.tile_col
                      << ") done -> " << base << "_gabor.png\n";
        }

        // 5) average the tile vectors -> ONE 24-number vector for this level
        if (used > 0) {
            levelCsv << level << "," << used;
            for (size_t i = 0; i < levelSum.size(); i++) levelCsv << "," << levelSum[i] / used;
            levelCsv << "\n";
        }
        std::cout << "  Tiles processed at this level: " << used << "\n";
    }

    std::cout << "\nDone. Open the 'output_gabor/' folder:\n"
              << "  gabor_kernels.png           the 8 stripe detectors themselves\n"
              << "  levelN_rX_cY_gabor.png      original tile + its 8 filter responses\n"
              << "  gabor_tile_features.csv     24 numbers for every tile\n"
              << "  gabor_level_features.csv    24 numbers per pyramid level (tiles averaged)\n";
    openslide_close(osr);
    return 0;
}
