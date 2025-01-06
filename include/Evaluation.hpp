#pragma once

#include <iomanip> // std::fixed, std::setprecision
#include <iostream>
#include <vector>

#include "DataLoader.hpp"
#include "Matrix.hpp"
#include "NN.hpp"

class Evaluation {

public:

	static double cross_validation(NN& nn, Dataset& dataset, std::size_t epochs, std::size_t batch_size, double lr, std::size_t k) {

		Dataset::shuffle(dataset.X, dataset.Y);

		std::cout << dataset.X.size() << std::endl;

		std::size_t fold_size = dataset.X.size() / k;

		std::vector<double> mse_folds(k, 0.0);

		for (std::size_t i = 0; i < k; ++i) {
			std::cout << "\n______ FOLD [" << i+1 << "/" << k << "] _____\n" <<  std::endl;
			std::size_t start = (i * fold_size);
			std::size_t end = (i+1 != k) ? ( (i+1) * fold_size - 1 ) : ( dataset.X.size() );

			// create new training set and test set
			Dataset training_set {};
			Dataset test_set {};
			for (std::size_t j = 0; j < dataset.X.size(); ++j) {

				const Matrix<double>& x = dataset.X.at(j);
				const Matrix<double>& y = dataset.Y.at(j);

				if (j >= start && j < end) { // test set
					test_set.X.push_back(x);
					test_set.Y.push_back(y);
				} else { // training set
					training_set.X.push_back(x);
					training_set.Y.push_back(y);
				}
			}

			// Train
			nn.train(training_set.X, training_set.Y, epochs, lr, batch_size);
			// Evaluate
			for (std::size_t j = 0; j < test_set.X.size(); ++j) {
				const Matrix<double>& x = dataset.X.at(j);
				const Matrix<double>& y = dataset.Y.at(j);
				Matrix<double> pred = nn.forward(x);
				mse_folds.at(i) += NN::mse(pred, y);
			}
			mse_folds.at(i) /= static_cast<double>(test_set.X.size());

			std::cout << "\n______ Fold loss = " << mse_folds.at(i) << " ______\n" << std::endl;
		}

		double cv_loss = 0.0;
		std::cout << "---------------\n";
		for (auto& e : mse_folds) {
			std::cout << e << std::endl;
			cv_loss += e;
		}
		std::cout << "---------------\n";
		cv_loss /= static_cast<double>(mse_folds.size());
		return cv_loss;

	}
};
