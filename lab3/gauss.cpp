#include <iostream>
#include <vector>
#include <cmath>  
using namespace std;

vector<vector<double>> makeGaussianKernel(int kernel_size, double sigma) {
    int r = kernel_size / 2;
    vector<vector<double>> kernel(kernel_size, vector<double>(kernel_size));

    double sum = 0.0;
    double sigma2 = sigma * sigma;
    double coeff = 1.0 / (2.0 * M_PI * sigma2);

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int x = i - r;  
            int y = j - r;
            double val = coeff * exp(-(x * x + y * y) / (2.0 * sigma2));
            kernel[i][j] = val;
            sum += val;
        }
    }

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}


vector<vector<int>> MyGaussianBlur(const vector<vector<int>>& img, int kernel_size, double sigma) {
    int h = img.size();
    if (h == 0) return img;
    int w = img[0].size();

    if (kernel_size % 2 == 0) {
        cerr << "kernel_size должен быть нечетным!" << endl;
        return img;
    }

    int r = kernel_size / 2; // радиус
    auto kernel = makeGaussianKernel(kernel_size, sigma);

    // Результат — отдельное изображение
    vector<vector<int>> out = img;
    for (int y = r; y < h - r; ++y) {
        for (int x = r; x < w - r; ++x) {
            double acc = 0.0;

            for (int ky = -r; ky <= r; ++ky) {
                for (int kx = -r; kx <= r; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;
                    acc += img[iy][ix] * kernel[ky + r][kx + r];
                }
            }

            int val = static_cast<int>(round(acc));
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            out[y][x] = val;
        }
    }

    return out;
}

void printMatrix(const vector<vector<int>>& m) {
    for (const auto& row : m) {
        for (int v : row) {
            cout << v << "\t";
        }
        cout << "\n";
    }
}

int main() {
    vector<vector<int>> img = {
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
        {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},
    };

    cout << "Исходная матрица:\n";
    printMatrix(img);

    int kernel_size = 3;
    double sigma = 1.0;

    auto blurred = MyGaussianBlur(img, kernel_size, sigma);

    cout << "\nПосле Gaussian Blur (kernel_size = " << kernel_size
         << ", sigma = " << sigma << "):\n";
    printMatrix(blurred);

    return 0;
}
