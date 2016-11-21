#include <iostream>
#include <random>

class Generator {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;
    float min;
    float max;
public:
    Generator(float mean, float stddev, float min, float max):
        distribution(mean, stddev), min(min), max(max)
    {}

    float operator ()() {
        while (true) {
            float number = this->distribution(generator);
            //if (number >= this->min && number <= this->max)
                return number;
        }
    }
Generator(float min, float max):
        distribution((min + max) / 2, (max - min) / 6), min(min), max(max)
    {}
};
