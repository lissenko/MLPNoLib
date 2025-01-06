# pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include "Matrix.hpp"

class NN {
private:
	std::size_t input_size = 0;
	std::size_t output_size = 0;
	bool activate_output = 0;
	bool classification = 0;
	std::size_t num_layers = 0;
	std::vector<std::size_t> layers {};
	std::vector<Matrix<double>> weights {};
	std::vector<Matrix<double>> derivatives {};
	std::vector<Matrix<double>> activations {};

public:
    NN() = default;

	NN(std::size_t input_size, std::vector<std::size_t> layers_p, std::size_t output_size, bool activate_output, bool classification)
	: input_size(input_size), output_size(output_size), activate_output(activate_output), classification(classification), layers(layers_p) {

		layers.insert(layers.begin(), input_size);
		layers.push_back(output_size);
		num_layers = layers.size();

		for (std::size_t i = 0; i < num_layers - 1; ++i) {
			derivatives.push_back( Matrix<double>::zeros(layers[i], layers[i+1]) );
		}

		for (std::size_t i = 0; i < num_layers; ++i) {
			activations.push_back( Matrix<double>::zeros(1, layers[i]) );
		}

	}

	void save_model(const std::string& file_path) {
		std::ofstream file(file_path, std::ios::out);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file for saving the model.");
		}

		file << input_size << " " << output_size << " " << activate_output << "\n";

		file << layers.size() << "\n";
		for (const auto& layer_size : layers) {
			file << layer_size << " ";
		}
		file << "\n";

		for (const auto& weight_matrix : weights) {
			file << weight_matrix.rows() << " " << weight_matrix.cols() << "\n";
			for (std::size_t i = 0; i < weight_matrix.rows(); ++i) {
				for (std::size_t j = 0; j < weight_matrix.cols(); ++j) {
					file << weight_matrix[i][j] << " ";
				}
				file << "\n";
			}
		}

		file.close();
	}

	void load_model(const std::string& file_path) {
		std::ifstream file(file_path, std::ios::in);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file for loading the model.");
		}
		file >> input_size >> output_size >> activate_output;
		std::size_t num_layers_loaded;
		file >> num_layers_loaded;
		layers.resize(num_layers_loaded);
		for (std::size_t i = 0; i < num_layers_loaded; ++i) {
			file >> layers[i];
		}
		num_layers = layers.size();
		weights.clear();
		derivatives.clear();
		activations.clear();

		for (std::size_t i = 0; i < num_layers - 1; ++i) {
			weights.push_back(Matrix<double>::zeros(layers[i], layers[i + 1]));
			derivatives.push_back(Matrix<double>::zeros(layers[i], layers[i + 1]));
		}
		for (std::size_t i = 0; i < num_layers; ++i) {
			activations.push_back(Matrix<double>::zeros(1, layers[i]));
		}
		for (std::size_t i = 0; i < weights.size(); ++i) {
			std::size_t rows, cols;
			file >> rows >> cols;
			for (std::size_t r = 0; r < rows; ++r) {
				for (std::size_t c = 0; c < cols; ++c) {
					file >> weights[i][r][c];
				}
			}
		}
		file.close();
	}


	double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	Matrix<double> sigmoid(const Matrix<double>& matrix) {
		Matrix<double> sigmoid_matrix = Matrix<double>::zeros(matrix.rows(), matrix.cols());
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			for (std::size_t j = 0; j < matrix.cols(); ++j) {
				sigmoid_matrix[i][j] = sigmoid(matrix[i][j]);
			}
		}
		return sigmoid_matrix;
	}

	Matrix<double> sigmoid_derivative(const Matrix<double>& matrix) {
		Matrix<double> sigmoid_derivative_matrix = Matrix<double>::zeros(matrix.rows(), matrix.cols());
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			for (std::size_t j = 0; j < matrix.cols(); ++j) {
				double a = matrix[i][j];
				sigmoid_derivative_matrix[i][j] = a * (1.0 - a);
			}
		}
		return sigmoid_derivative_matrix;
	}

	double relu(double x) {
		return std::max(0.0, x);
	}

	Matrix<double> relu(const Matrix<double>& matrix) {
		Matrix<double> relu_matrix = Matrix<double>::zeros(matrix.rows(), matrix.cols());
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			for (std::size_t j = 0; j < matrix.cols(); ++j) {
				relu_matrix[i][j] = relu(matrix[i][j]);
			}
		}
		return relu_matrix;
	}

	Matrix<double> relu_derivative(const Matrix<double>& matrix) {
		Matrix<double> relu_derivative_matrix = Matrix<double>::zeros(matrix.rows(), matrix.cols());
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			for (std::size_t j = 0; j < matrix.cols(); ++j) {
				double a = matrix[i][j];
				relu_derivative_matrix[i][j] = (a <= 0.0) ? 0.0 : 1;
			}
		}
		return relu_derivative_matrix;
	}

	Matrix<double> softmax(const Matrix<double>& matrix) {
		Matrix<double> exp_matrix = Matrix<double>::zeros(matrix.rows(), matrix.cols());
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			double row_sum = 0.0;
			for (std::size_t j = 0; j < matrix.cols(); ++j) {
				exp_matrix[i][j] = std::exp(matrix[i][j]);
				row_sum += exp_matrix[i][j];
			}
			for (std::size_t j = 0; j < matrix.cols(); ++j) {
				exp_matrix[i][j] /= row_sum;
			}
		}
		return exp_matrix;
	}

	Matrix<double> forward(const Matrix<double>& input) {
		if (input.cols() != input_size) {
            throw std::invalid_argument("The input size is different from the network input size.");
		}
		activations.at(0) = input;
		for (std::size_t i = 0; i < num_layers - 1; ++i) {
			Matrix<double> z = activations.at(i) * weights.at(i);
			if (i == num_layers-2 && classification) {
				activations.at(i + 1) = softmax(z);
			} else if (i == num_layers-2 && !activate_output && !classification) { // last layer, prediction and not activate
				activations.at(i+1) = z;
			} else {
				activations.at(i+1) = relu(z); // TODO
			}
		}
		return activations.at(num_layers-1); // output
	}

	void backward(Matrix<double> error) {
		for (std::size_t i = num_layers - 1; i > 0; --i) {
			Matrix<double> activation = activations.at(i);
			if (i == num_layers-1) { // last layer
				if (activate_output && !classification) {
					error = error.elementWiseMultiply(relu_derivative(activation)); // TODO
				}
				// Warning: if classification and softmax, do nothing because dE/dz = pred - target
			} else {
				error = error.elementWiseMultiply(relu_derivative(activation));
			}
			derivatives.at(i - 1) = derivatives.at(i - 1) + (activations.at(i - 1).transpose() * error);
			error = error * weights.at(i - 1).transpose();
		}
	}

	 void gradient_descent(double lr, double batch_scaling_factor) {

		for (Matrix<double>& derivative : derivatives) {
			derivative = derivative / batch_scaling_factor;
		}

	 	for (std::size_t i = 0; i < num_layers - 1; ++i) {
	 		weights[i] = weights[i] - (derivatives[i] * lr);
	 	}
		// Reset derivatives
		for (Matrix<double>& derivative : derivatives) {
			derivative.fill(0.0);
		}
	 }

	static double mse(const Matrix<double>& output, const Matrix<double>& target) {
		Matrix<double> error = output - target;
		return (error.elementWiseMultiply(error)).sum() / static_cast<double>(error.cols());
	}

	static int is_prediction_correct(const Matrix<double>& output, const Matrix<double>& target) {
		// the index of the maximum value in the output (predicted class)
		std::size_t predicted_class = 0;
		double max_value = output[0][0];
		for (std::size_t i = 1; i < output.cols(); ++i) {
			if (output[0][i] > max_value) {
				max_value = output[0][i];
				predicted_class = i;
			}
		}

		// the index of the maximum value in the target (true class)
		std::size_t true_class = 0;
		max_value = target[0][0];
		for (std::size_t i = 1; i < target.cols(); ++i) {
			if (target[0][i] > max_value) {
				max_value = target[0][i];
				true_class = i;
			}
		}
		return (predicted_class == true_class) ? 1 : 0;
	}
	
	void train(std::vector<Matrix<double>>& X, std::vector<Matrix<double>>& Y, std::size_t epochs, double lr, std::size_t batch_size) {
		// reset weights
		std::cout << "\n____ Weights Init ____\n\n";
		weights.clear();
		for (std::size_t i = 0; i < num_layers - 1; ++i) {
			weights.push_back( Matrix<double>::glorot_init(layers[i], layers[i+1]) );
		}

		for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
			std::cout << "Epoch: [" << epoch+1 << "/" << epochs << "]";
			// double total_mse = 0.0;
			int total_correct = 0.0;

			for (std::size_t i = 0; i < X.size(); i += batch_size) {
				// Before a batch
				for (std::size_t j = i; j < X.size() && j < i + batch_size; ++j) {
					Matrix<double> output = forward(X.at(j));
					Matrix<double> target = Y.at(j);
					Matrix<double> error = output - target;
					// total_mse += mse(output, target);
					total_correct += is_prediction_correct(output, target);
					backward(error);
				}
				// After a batch
				gradient_descent(lr, static_cast<double>(batch_size));
			}

			// std::cout << " -> loss = " << total_mse / static_cast<double>(X.size());
			std::cout << " -> correctly classified: " << (static_cast<double>(total_correct) / static_cast<double>(X.size())) * 100.0 << "%";
			std::cout << std::endl;
		}
	}

};


