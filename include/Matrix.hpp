#pragma once

#include <iostream>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

template <typename T>
class Matrix {

private:
	std::size_t n = 0;
	std::size_t m = 0;
	std::vector<std::vector<T>> matrix {};

public:
	Matrix(std::vector<std::vector<T>> matrix)
	: n(matrix.size()), m(matrix.at(0).size()), matrix(matrix)  {
		for (auto row : matrix) {
			if (row.size() != m) {
				throw std::invalid_argument("Matrix constructor error: Inconsistent size.");
			}
		}
	}

    Matrix(std::initializer_list<std::vector<T>> init_list) : Matrix(std::vector<std::vector<T>>(init_list)) {}

	Matrix<T> transpose() {
        std::vector<std::vector<T>> transposed_matrix(m, std::vector<T>(n, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				transposed_matrix[j][i] = matrix[i][j];
			}
		}
		return Matrix<T>(transposed_matrix);
	}

    const std::vector<T>& operator[](std::size_t index) const { return matrix[index]; }
    std::vector<T>& operator[](std::size_t index) { return matrix[index]; }
    std::size_t rows() const { return n; }
    std::size_t cols() const { return m; }

    Matrix<T> operator*(const Matrix<T>& B) const {
        if (m != B.n) {
			std::cerr << "A.shape = " << "(" << n << "x" << m << ")\n";
			std::cerr << "B.shape = " << "(" << B.n << "x" << B.m << ")\n";
            throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
        }
        std::vector<std::vector<T>> C(n, std::vector<T>(B.m, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < B.m; ++j) {
				for (std::size_t k = 0; k < m; ++k) {
					C[i][j] += matrix[i][k] * B[k][j];
				}
			}
		}
		return Matrix(C);
	}

    Matrix<T> operator*(const double d) const {
        std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				C[i][j] = matrix[i][j] * d;
			}
		}
		return Matrix(C);
	}

    Matrix<T> operator/(const double d) const {
        std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				C[i][j] = matrix[i][j] / d;
			}
		}
		return Matrix(C);
	}

    Matrix<T> operator-(const Matrix<T>& B) const {
        if (n != B.n || m != B.m) {
			std::cerr << "A.shape = " << "(" << n << "x" << m << ")\n";
			std::cerr << "B.shape = " << "(" << B.n << "x" << B.m << ")\n";
            throw std::invalid_argument("Matrix dimensions do not allow subtraction.");
        }
        std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				C[i][j] = matrix[i][j] - B[i][j];
			}
		}
		return Matrix<T>(C);
	}

    Matrix<T> operator+(const Matrix<T>& B) const {
        if (n != B.n || m != B.m) {
			std::cerr << "A.shape = " << "(" << n << "x" << m << ")\n";
			std::cerr << "B.shape = " << "(" << B.n << "x" << B.m << ")\n";
            throw std::invalid_argument("Matrix dimensions do not allow addition.");
        }
        std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				C[i][j] = matrix[i][j] + B[i][j];
			}
		}
		return Matrix<T>(C);
	}

	Matrix<T> elementWiseMultiply(const Matrix<T>& B) const {
		if (n != B.n || m != B.m) {
			std::cerr << "A.shape = " << "(" << n << "x" << m << ")\n";
			std::cerr << "B.shape = " << "(" << B.n << "x" << B.m << ")\n";
			throw std::invalid_argument("Matrix dimensions do not allow element-wise multiplication.");
		}
		std::vector<std::vector<T>> C(n, std::vector<T>(m, 0));
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				C[i][j] = matrix[i][j] * B[i][j];
			}
		}
		return Matrix<T>(C);
	}

	static T min(const Matrix<T>& matrix, const std::size_t column_idx) {
		T min_value = std::numeric_limits<T>::max();
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			if (matrix[i][column_idx] < min_value) {
				min_value = matrix[i][column_idx];
			}
		}
		return min_value;
	}

	static T max(const Matrix<T>& matrix, const std::size_t column_idx) {
		T max_value = std::numeric_limits<T>::min();
		for (std::size_t i = 0; i < matrix.rows(); ++i) {
			if (matrix[i][column_idx] > max_value) {
				max_value = matrix[i][column_idx];
			}
		}
		return max_value;
	}

	void fill(double d) {
        std::vector<std::vector<T>> new_matrix(n, std::vector<T>(m, d));
		matrix = new_matrix;
	}

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
        os << "[\n";
        for (const auto& row : mat.matrix) {
            os << "  [";
            for (std::size_t j = 0; j < row.size(); ++j) {
                os << row[j];
                if (j < row.size() - 1) {
                    os << ", ";
                }
            }
            os << "]\n";
        }
        os << "]\n";
        return os;
    }

	static Matrix<T> zeros(std::size_t n, std::size_t m) {
		std::vector<std::vector<T>> matrix {};
		for (std::size_t i = 0; i < n; ++i) {
			matrix.push_back(std::vector<T>(m, T(0.0)));
		}
		return Matrix<T>(matrix);
	}

	T sum() {
		T res {};
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < m; ++j) {
				res += matrix[i][j];
			}
		}
		return res;
	}

	std::size_t vecArgMax() {
		std::vector<T>& v = matrix.at(0);
		typename std::vector<T>::iterator max = max_element(v.begin(), v.end());
		return std::distance(v.begin(), max);
	}

	static Matrix<T> glorot_init(std::size_t n_in, std::size_t n_out) {
		std::random_device rd;
		std::mt19937 gen(rd());

		double limit = std::sqrt( 6.0 / (static_cast<double>(n_in) + static_cast<double>(n_out)) );
		std::uniform_real_distribution<> dis(-limit, limit);

		std::vector<std::vector<T>> matrix {};
        for (std::size_t i = 0; i < n_in; ++i) {
            std::vector<T> row;
            for (std::size_t j = 0; j < n_out; ++j) {
                row.push_back(dis(gen));
            }
            matrix.push_back(row);
        }
		return Matrix<T>(matrix);
	}

	static Matrix<T> random_uniform_init(std::size_t n_in, std::size_t n_out, double range = 0.1) {
		if (n_in == 0 || n_out == 0) {
			throw std::invalid_argument("n_in and n_out must be greater than 0");
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(-range, range);

		std::vector<std::vector<T>> matrix(n_in, std::vector<T>(n_out));
		for (std::size_t i = 0; i < n_in; ++i) {
			for (std::size_t j = 0; j < n_out; ++j) {
				matrix[i][j] = dis(gen);
			}
		}

		return Matrix<T>(matrix);
	}


};
