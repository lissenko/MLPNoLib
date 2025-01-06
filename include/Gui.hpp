# pragma once

#include <SFML/Graphics.hpp>
#include <vector>

#include "Matrix.hpp"

const int grid_size = 28;
const int cell_size = 20;

void draw_with_mouse(Matrix<double>& mnist_sample) {
    const int window_size = grid_size * cell_size;
    sf::RenderWindow window(sf::VideoMode(window_size, window_size), "Draw a Digit");
    std::vector<std::vector<int>> grid(grid_size, std::vector<int>(grid_size, 0));
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
                int x = mouse_pos.x / cell_size;
                int y = mouse_pos.y / cell_size;
                if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
                    grid[y][x] = 255;
                }
            }
        }
        window.clear(sf::Color::White);
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                sf::RectangleShape cell(sf::Vector2f(cell_size - 1, cell_size - 1));
                cell.setPosition(static_cast<float>(j) * cell_size, static_cast<float>(i) * cell_size);
                int intensity = grid[i][j];
                cell.setFillColor(sf::Color(255 - static_cast<sf::Uint8>(intensity), 255 - static_cast<sf::Uint8>(intensity), 255 - static_cast<sf::Uint8>(intensity)));
                window.draw(cell);
            }
        }

        window.display();
    }
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            mnist_sample[0][i * grid_size + j] = static_cast<double>(grid[i][j]);
        }
    }
}
