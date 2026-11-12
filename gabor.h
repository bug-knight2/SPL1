// gabor.h
// ---------------------------------------------------------------------
// The HEART of the project: a 100% hand-written Gabor filter.
// Nothing in this file uses OpenCV, skimage, or any vision library.
// It only uses the C++ standard library (vector, cmath).
//
// A Gabor filter is a small "stripe detector":
//   - it is a tiny picture (called a KERNEL) made of fuzzy stripes
//   - we slide it over the tile and, at every pixel, measure
//     "how much do the pixels around here look like my stripes?"
//   - big number  = the tile has stripes like mine at this spot
//     small number = no such stripes here
// We make 8 different stripe detectors (4 directions x 2 stripe widths)
// and use how strongly each one fires as the texture features.
// ---------------------------------------------------------------------
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

constexpr double kPi = 3.14159265358979323846;

// The 5 knobs that control how the stripes look.
struct GaborParams {
    double lambda; // wavelength: distance (in pixels) from one stripe to the next
    double theta;  // direction of the wave, in RADIANS (0 = wave runs left->right)
    double sigma;  // size of the Gaussian "fade-out" bubble around the stripes
    double gamma;  // how squashed the bubble is (0.5 = stretched along the stripes)
    double psi;    // phase shift of the wave (0 = a bright stripe in the centre)
};

// A kernel is just a small square grid of numbers.
struct Kernel {
    int size = 0;           // side length of the square (always an odd number)
    std::vector<double> w;  // size*size numbers, stored row by row
};

// ---------------------------------------------------------------------
// STEP A: build one Gabor kernel from the formula
//
//   g(x,y) = exp( -(x'^2 + gamma^2 * y'^2) / (2 sigma^2) ) * cos( 2*pi*x'/lambda + psi )
//
//   x' =  x*cos(theta) + y*sin(theta)      <- distance ALONG the wave direction
//   y' = -x*sin(theta) + y*cos(theta)      <- distance ACROSS it
//
// x, y are measured from the CENTRE of the kernel.
// ---------------------------------------------------------------------
inline Kernel makeGaborKernel(const GaborParams& p) {
    // How big must the grid be? The Gaussian bubble has faded to almost
    // nothing after 3 sigma. If gamma < 1 the bubble is stretched by
    // 1/gamma in one direction, so we need a bigger grid for that.
    double stretch = std::max(1.0, 1.0 / p.gamma);
    int half = static_cast<int>(std::ceil(3.0 * p.sigma * stretch));

    Kernel k;
    k.size = 2 * half + 1;                         // e.g. half=14 -> size 29
    k.w.assign(static_cast<size_t>(k.size) * k.size, 0.0);

    double c = std::cos(p.theta);                  // computed once, used for every cell
    double s = std::sin(p.theta);
    double sum = 0.0;                              // we will need the average value later

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            // rotate the (x,y) position by theta
            double xr =  x * c + y * s;            // x'
            double yr = -x * s + y * c;            // y'

            // the fade-out bubble (Gaussian): 1 in the centre, ->0 far away
            double bubble = std::exp(-(xr * xr + p.gamma * p.gamma * yr * yr)
                                     / (2.0 * p.sigma * p.sigma));

            // the stripes (cosine wave): goes bright, dark, bright, dark ...
            double stripes = std::cos(2.0 * kPi * xr / p.lambda + p.psi);

            double value = bubble * stripes;       // fuzzy stripes!
            k.w[static_cast<size_t>(y + half) * k.size + (x + half)] = value;
            sum += value;
        }
    }

    // Make the kernel "zero-mean": subtract the average value from every
    // cell. Why? A perfectly flat grey area has no stripes, so it should
    // give an answer of 0. Without this step, a bright flat area would give
    // a big answer just because it is bright, which is not texture.
    double mean = sum / (static_cast<double>(k.size) * k.size);
    for (auto& v : k.w) v -= mean;

    return k;
}

// ---------------------------------------------------------------------
// STEP B: RAW 2D filtering (convolution)
//
// For every pixel of the image we:
//   1. lay the kernel centre on that pixel
//   2. multiply each kernel number by the image pixel under it
//   3. add all those products up -> that sum is the answer for the pixel
//
// Near the image border the kernel hangs outside the picture. We handle
// that by "clamping": a pixel outside the picture is replaced by the
// nearest pixel that is inside (the edge pixel is just repeated).
//
// Note: strictly this is "correlation", but a Gabor kernel looks the same
// after turning it 180 degrees, so correlation == convolution here.
// ---------------------------------------------------------------------
inline std::vector<double> convolve2D(const std::vector<uint8_t>& img,
                                      int w, int h, const Kernel& k) {
    std::vector<double> out(static_cast<size_t>(w) * h, 0.0);
    int half = k.size / 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double sum = 0.0;
            for (int ky = -half; ky <= half; ky++) {
                // clamp the row so we never read outside the image
                int yy = std::min(std::max(y + ky, 0), h - 1);
                const uint8_t* row = &img[static_cast<size_t>(yy) * w];
                const double* krow = &k.w[static_cast<size_t>(ky + half) * k.size];
                for (int kx = -half; kx <= half; kx++) {
                    // clamp the column too
                    int xx = std::min(std::max(x + kx, 0), w - 1);
                    sum += row[xx] * krow[kx + half];
                }
            }
            out[static_cast<size_t>(y) * w + x] = sum;
        }
    }
    return out;
}

// ---------------------------------------------------------------------
// STEP C: turn a whole response image into 3 summary numbers
//
//   mean     = average of |response|      "how strong are the stripes, overall?"
//   variance = spread of |response|       "are they steady or patchy?"
//   energy   = average of response^2      "total stripe power"
//
// We ignore a border strip of width 'margin' because there the kernel was
// hanging off the tile and the numbers are not trustworthy.
// ---------------------------------------------------------------------
struct ResponseStats {
    double mean = 0, variance = 0, energy = 0;
};

inline ResponseStats computeStats(const std::vector<double>& resp,
                                  int w, int h, int margin) {
    ResponseStats st;
    double sumAbs = 0, sumAbsSq = 0, sumSq = 0;
    long long n = 0;
    for (int y = margin; y < h - margin; y++) {
        for (int x = margin; x < w - margin; x++) {
            double r = resp[static_cast<size_t>(y) * w + x];
            double a = std::fabs(r);
            sumAbs   += a;
            sumAbsSq += a * a;
            sumSq    += r * r;
            n++;
        }
    }
    if (n == 0) return st;                     // tile too small: all zeros
    st.mean     = sumAbs / n;
    st.energy   = sumSq / n;
    // variance = E[a^2] - (E[a])^2
    st.variance = sumAbsSq / n - st.mean * st.mean;
    return st;
}
