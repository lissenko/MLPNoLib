# pragma once

#include <algorithm>
#include <ctime>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <iostream>
#include <limits>
#include <vector>

#include "Matrix.hpp"


struct Dataset {
    std::vector<Matrix<double>> X {};
    std::vector<Matrix<double>> Y {};

    std::vector<double> X_min_values {};
    std::vector<double> X_max_values {};
	double Y_min = std::numeric_limits<double>::max();
	double Y_max = std::numeric_limits<double>::min();

	void find_features_min_max() {
		std::size_t feature_number = X.at(0).cols();
		for (std::size_t j = 0; j < feature_number; ++j) { // pour chaque feature, trouver min et max
			X_min_values.push_back( std::numeric_limits<double>::max() );
			X_max_values.push_back( std::numeric_limits<double>::min() );
			for (std::size_t i = 0; i < X.size(); ++i) { // Pour chaque sample
				double val = X.at(i)[0][j];
				if (val < X_min_values.at(j)) { X_min_values.at(j) = val; }
				if (val > X_max_values.at(j)) { X_max_values.at(j) = val; }
			}
		}
	}

    void normalize_features() {
		std::cout << "\n___ Normalizing features ___\n";
		find_features_min_max();
		std::size_t feature_number = X.at(0).cols();
		for (std::size_t j = 0; j < feature_number; ++j) { // pour chaque feature
			double min = X_min_values.at(j);
			double max = X_max_values.at(j);
			for (std::size_t i = 0; i < X.size(); ++i) { // Pour chaque sample
				double& x = X.at(i)[0][j];
				x = (x - min) / (max - min);
			}
		}

    }

	void find_target_min_max() {
		for (std::size_t i = 0; i < Y.size(); ++i) {
			double val = Y.at(i)[0][0];
			if (val < Y_min) { Y_min = val; }
			if (val > Y_max) { Y_max = val; }
		}
	}

	void normalize_target() {
		find_target_min_max();
		for (std::size_t i = 0; i < Y.size(); ++i) {
			double& x = Y.at(i)[0][0];
			x = (x - Y_min) / (Y_max - Y_min);
		}
	}

	void one_hot_encode_target(std::size_t num_classes) {
		std::cout << "\n___ One hot encoding targets ___\n";
		std::vector<Matrix<double>> one_hot_Y {};
		for (std::size_t i = 0; i < Y.size(); ++i) {
			Matrix<double> one_hot_vector = Matrix<double>::zeros(1, num_classes);
			one_hot_vector[0][static_cast<std::size_t>(Y.at(i)[0][0])] = 1.0;
			one_hot_Y.push_back(one_hot_vector);
		}
		Y = one_hot_Y;
	}

	Matrix<double> normalize_sample(const Matrix<double>& sample) {
		std::size_t feature_number = sample.cols();
		std::vector<double> C(feature_number, 0);
		for (std::size_t j = 0; j < feature_number; ++j) { // pour chaque feature
			double min = X_min_values.at(j);
			double max = X_max_values.at(j);
			double x = sample[0][j];
			C[j] = (x - min) / (max - min);
		}
		return Matrix<double>( {C} );

	}

	Matrix<double> denormalize_sample(const Matrix<double>& sample) {
		std::vector<double> C(sample.cols(), 0);
		for (std::size_t j = 0; j < sample.cols(); ++j) {
			double min = X_min_values.at(j);
			double max = X_max_values.at(j);
			double x = sample[0][j];
			C.at(j) = (x * (max-min)) + min;
		}
		return Matrix<double>( {C} );
	}

	Matrix<double> denormalize_target(const Matrix<double>& target) {
		std::vector<double> C(1, 0);
		C[0] = (target[0][0] * (Y_max - Y_min)) + Y_min;
		return Matrix<double>( {C} );
	}

	static void shuffle(std::vector<Matrix<double>>& features, std::vector<Matrix<double>>& target) {
		if (features.size() != target.size()) {
			throw std::invalid_argument("Features and target size mismatch.");
		}
		std::size_t n = features.size();
		std::vector<std::size_t> indices(n);
		for (std::size_t i = 0; i < n; ++i) {
			indices[i] = i;
		}
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(indices.begin(), indices.end(), g);

		std::vector<Matrix<double>> shuffled_features {};
		std::vector<Matrix<double>> shuffled_target {};

		for (std::size_t i = 0; i < n; ++i) {
			shuffled_features.push_back(features[indices[i]]);
			shuffled_target.push_back(target[indices[i]]);
		}
		features = std::move(shuffled_features);
		target = std::move(shuffled_target);
	}

};


int readInt(std::ifstream &file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<Matrix<double>> readImages(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    int magicNumber = readInt(file);
    if (magicNumber != 2051) { // 0x00000803
        throw std::runtime_error("Invalid magic number in image file: " + std::to_string(magicNumber));
    }
	std::size_t numImages = static_cast<std::size_t>(readInt(file));
    int numRows = readInt(file);
    int numCols = readInt(file);
	std::size_t imageSize = static_cast<std::size_t>(numRows * numCols);
    std::vector<Matrix<double>> matrix_images {};
    for (std::size_t i = 0; i < numImages; ++i) {
        std::vector<double> image(imageSize);
        for (std::size_t j = 0; j < imageSize; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<double>(pixel); // Directly convert to double
        }
        matrix_images.push_back({image});
    }
    return matrix_images;
}

std::vector<Matrix<double>> readLabels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    int magicNumber = readInt(file);
    if (magicNumber != 2049) { // 0x00000801
        throw std::runtime_error("Invalid magic number in label file: " + std::to_string(magicNumber));
    }
    int numLabels = readInt(file);
    std::vector<Matrix<double>> labelMatrices {};
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labelMatrices.push_back({{static_cast<double>(label)}});
    }
    return labelMatrices;
}

class DataLoader {

private:
public:

	static Dataset arithmetic_sum(const std::size_t num_items, double max_val) {
		std::srand(static_cast<unsigned int>(std::time(nullptr)));
		std::vector<Matrix<double>> X {};
		std::vector<Matrix<double>> Y {};
		for (std::size_t i = 0; i < num_items; ++i) {
			double a = static_cast<double>(std::rand() % static_cast<int>((max_val + 1)));
			double b = static_cast<double>(std::rand() % static_cast<int>((max_val + 1)));
			double c = static_cast<double>(std::rand() % static_cast<int>((max_val + 1)));
			X.push_back({ { a, b, c } });
			Y.push_back({ { a + b + c } });
		}
		return {X, Y};
	}

	static Dataset load_mnist(const std::string& imagesFile, const std::string& labelsFile) {
		std::vector<Matrix<double>> images = readImages(imagesFile);
        std::vector<Matrix<double>> labels = readLabels(labelsFile);
		return {images, labels};
	}

    static Dataset load_dataset(const std::string& csv_file) {
        std::vector<Matrix<double>> X;
        std::vector<Matrix<double>> Y;
        std::ifstream file(csv_file);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + csv_file);
        }
        std::string line;
        // skip the header
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<double> row_data;
            while (std::getline(ss, value, ',')) {
                row_data.push_back(std::stod(value));
            }
            if (row_data.size() < 2) {
                throw std::runtime_error("Invalid row in CSV file: " + line);
            }
            // target
            double target = row_data[0];
            std::vector<double> features(row_data.begin() + 1, row_data.end());
            X.emplace_back(Matrix<double>({features}));
            Y.emplace_back(Matrix<double>({{target}}));
        }
        file.close();
        return {X, Y};
    }

};
