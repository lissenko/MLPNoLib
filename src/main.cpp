#include <iostream>

#include "DataLoader.hpp"
#include "Evaluation.hpp"
#include "Gui.hpp"
#include "Matrix.hpp"
#include "NN.hpp"

int main() {

	bool train = 0;
	bool test = 0;

	if (train) {
		// Dataset dataset = DataLoader::arithmetic_sum(1000, 1000.0);
		// Dataset dataset = DataLoader::load_dataset("./datasets/BostonHousing.csv");
		Dataset train_dataset = DataLoader::load_mnist("./datasets/mnist/train-images.idx3-ubyte", "./datasets/mnist/train-labels.idx1-ubyte");
		train_dataset.normalize_features();
		train_dataset.one_hot_encode_target(10);
		// dataset.normalize_target();

		bool activate_output = 0;
		bool classification = 1;
		std::size_t input_size = train_dataset.X.at(0).cols();
		std::size_t epochs = 20;
		std::size_t batch_size = 64;
		double lr = 0.1;

		NN network(input_size, {128, 64, 32}, 10, activate_output, classification);
		network.train(train_dataset.X, train_dataset.Y, epochs, lr, batch_size);
		network.save_model("models/final_mnist.model");

	} else if (test) {
		NN network {};
		network.load_model("models/final_mnist.model");
		Dataset test_dataset = DataLoader::load_mnist("./datasets/mnist/t10k-images.idx3-ubyte", "./datasets/mnist/t10k-labels.idx1-ubyte");
		test_dataset.normalize_features();
		test_dataset.one_hot_encode_target(10);

		int correct = 0;
		for (std::size_t i = 0; i < test_dataset.X.size(); ++i) {
			Matrix<double> input = test_dataset.X.at(i);
			Matrix<double> target = test_dataset.Y.at(i);
			Matrix<double> vector_pred = network.forward(input);
			correct += NN::is_prediction_correct(vector_pred, target);
		}
		std::cout << "correctly classified: " << (static_cast<double>(correct) / test_dataset.X.size()) * 100.0 << "%\n";

	} else {
		NN network {};
		network.load_model("models/final_mnist.model");

		Matrix<double> mnist_sample = Matrix<double>::zeros(1, grid_size * grid_size);

		for (;;) {
			draw_with_mouse(mnist_sample);
			mnist_sample = mnist_sample / 255.0;
			Matrix<double> vector_pred = network.forward(mnist_sample);
			std::cout << vector_pred.vecArgMax() << std::endl;
		}
	}

    return 0;
}

