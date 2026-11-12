# Quantitative Texture Analysis of H&E-Stained Histology Images Using Handcrafted Feature Descriptors

A C++ project that extracts **handcrafted texture features** from H&E-stained histology slides (`.svs`), implemented **from scratch**: no OpenCV, no scikit-image, no pre-built feature-extraction libraries. The goal is an objective, reproducible numerical description of tissue texture that can later feed a cancer-diagnosis classifier.

> **Author:** Md Daud Islam Rafi (Roll: 1726)
> **Supervisor:** Dr. Mohammad Shoyaib
> **Course project:** SPL-1

---

## Project status

| Method | Status |
|---|---|
| SVS reading, grayscale conversion, tiling | Done |
| Local Binary Patterns (LBP) | Done (hand-written) |
| Gabor filter bank | Done (hand-written) |
| Grey-Level Co-occurrence Matrix (GLCM) | Planned (after midterm) |
| Histogram of Oriented Gradients (HOG) | Planned (after midterm) |
| Wavelet method | Planned (after midterm) |
| Concatenate all features into one vector per tile | Planned |

## Pipeline

```
.svs slide -> OpenSlide (read pyramid level) -> grayscale -> tiles
           -> skip blank background tiles -> feature extraction per tile
           -> feature vector (CSV)
```

1. **Read** the slide with OpenSlide (used only for decoding the `.svs` file).
2. **Grayscale**: `0.299 R + 0.587 G + 0.114 B`, written as a raw pixel loop.
3. **Tiling**: the level is divided into square tiles (ceiling division, so edge tiles may be smaller).
4. **Background rejection**: tiles that are almost white are skipped.
5. **Features** are computed per tile with the hand-written methods below.

## Implemented methods

### Local Binary Patterns (LBP)
Each pixel is compared with its 8 neighbours; every neighbour that is >= the centre contributes one bit, producing an 8-bit code (0-255). The tile's **256-bin histogram** of these codes is its LBP feature vector.

- Output: LBP image (PNG) + 256-bin histogram (CSV) per tile.

### Gabor filter bank
Eight hand-built Gabor kernels (2 wavelengths x 4 orientations: 0, 45, 90, 135 degrees) are convolved with each tile using a raw 2D convolution with border clamping. For every filter response, three statistics are computed (mean of |response|, variance, energy), giving **8 x 3 = 24 features per tile**.

- Output: kernel picture, response montage per tile, per-tile and per-level feature CSVs.
- Includes a `--selftest` that feeds synthetic stripe images in 4 directions and checks that the matching filter responds most strongly. It needs no slide.

## Repository structure

```
SPL1/
├── README.md
├── .gitignore
├── include/
│   ├── gabor.h               # hand-written Gabor kernel, convolution, statistics
│   └── stb_image_write.h     # single-header PNG writer (third-party, only saves images)
├── src/
│   ├── lbp_main.cpp          # slide -> grayscale -> tiles -> LBP
│   └── gabor_main.cpp        # slide -> tiles (all levels) -> Gabor features
├── docs/                     # proposal and notes
└── results/                  # a few sample output images
```

GLCM, HOG and Wavelet will be added as new files in `include/` and `src/` as they are completed.

## Requirements

- Linux (Ubuntu) or WSL
- `g++` with C++17 support
- OpenSlide and `pkg-config`

```bash
sudo apt-get update
sudo apt-get install -y g++ pkg-config libopenslide-dev
```

## Build

```bash
# LBP
g++ -std=c++17 -O2 -Iinclude src/lbp_main.cpp $(pkg-config --cflags --libs openslide) -o lbp_demo

# Gabor
g++ -std=c++17 -O2 -Iinclude src/gabor_main.cpp $(pkg-config --cflags --libs openslide) -o gabor_demo
```

## Run

```bash
# LBP:  ./lbp_demo <slide.svs> [tileSize=128] [maxTiles=8]
./lbp_demo my_slide.svs 128 8

# Gabor: ./gabor_demo <slide.svs> [tileSize=256] [maxTilesPerLevel=6] [whiteThreshold=220]
./gabor_demo my_slide.svs 256 6 220

# Gabor self-test (no slide needed)
./gabor_demo --selftest
```

## Output

| Program | Folder | Contents |
|---|---|---|
| `lbp_demo` | `output/` | full grayscale level, per-tile original + LBP image, 256-bin histogram CSV |
| `gabor_demo` | `output_gabor/` | kernel montage, per-tile response montage, `gabor_tile_features.csv`, `gabor_level_features.csv` |
| `gabor_demo --selftest` | `output_gabor_selftest/` | kernel montage and response pictures for synthetic stripes |

Slides (`.svs`) and generated output folders are not stored in this repository (see `.gitignore`).

## Planned final feature vector

For each tile, the outputs of all five methods will be concatenated into a single vector (LBP + GLCM + Gabor + HOG + Wavelet) to give a compact numerical representation of its texture and morphology for downstream classification.

## Third-party code

- [OpenSlide](https://openslide.org/): reading `.svs` files only.
- [stb_image_write.h](https://github.com/nothings/stb): writing PNG files only.

All feature-extraction logic (LBP, Gabor, and the upcoming methods) is written from scratch.
