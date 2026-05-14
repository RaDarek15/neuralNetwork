#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include "json.hpp"

using json = nlohmann::json;

//-------------------CLASSES-------------------

class Warstwa
{
private:
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    double static sigmoid(double z)
    {
        return 1.0 / (1.0 + exp(-z));
    }
    std::vector<double> forwardPass(
        std::vector<double> inputy)
    {
        std::vector<double> wyniki;

        for (int i = 0; i < weights.size(); i++)
        {
            double wynik = 0;

            for (int j = 0; j < inputy.size(); j++)
            {
                wynik += inputy[j] * weights[i][j];
            }

            wynik += biases[i];

            wyniki.push_back(sigmoid(wynik));
        }

        return wyniki;
    }
    Warstwa(int neuronCount, int inputCount);
    Warstwa(std::vector<std::vector<double>> weights, std::vector<double> biases);
};
Warstwa::Warstwa(int neuronCount, int inputCount)
{
    weights.resize(neuronCount);
    for (auto &i : weights)
        i.resize(inputCount);
    biases.resize(neuronCount);
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < neuronCount; i++)
    {
        for (int j = 0; j < inputCount; j++)
        {
            weights[i][j] = dist(mt);
        }
    }
    for (int i = 0; i < neuronCount; i++)
    {
        biases[i] = dist(mt);
    }
}
Warstwa::Warstwa(std::vector<std::vector<double>> weights, std::vector<double> biases) : biases(biases), weights(weights) {};

//-------------------DIVIDER-------------------

class siec
{
private:
public:
    std::vector<Warstwa> layers;

    std::vector<double> forwardPass(std::vector<double> inputs)
    {
        std::vector<double> layerOutput = inputs;
        for (int i = 0; i < layers.size(); i++)
        {
            layerOutput = layers[i].forwardPass(layerOutput);
        }
        return layerOutput;
    }
    double srStrata(std::vector<std::vector<double>> wynikSieci, std::vector<std::vector<double>> expectedValues)
    {
        double srStrata;
        int i = 0;
        for (; i < expectedValues.size(); i++)
        {
            for (int j = 0; j < expectedValues[i].size(); j++)
            {
                srStrata += pow((expectedValues[i][j] - wynikSieci[i][j]), 2);
            }
        }
        return srStrata / expectedValues.size();
    }

    std::vector<std::vector<double>> biasNumGrad(double increment, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expectedValues)
    {
        std::vector<std::vector<double>> wynikiSieci;
        for (int i = 0; i < inputs.size(); i++)
        {
            wynikiSieci.push_back(forwardPass(inputs[i]));
        }

        double ogStrata = srStrata(wynikiSieci, expectedValues);
        wynikiSieci.clear();
        std::vector<double> layerGradient;
        std::vector<std::vector<double>> gradient;

        for (int i = 0; i < layers.size(); i++)
        {
            const std::vector<double> constBiases = layers[i].biases;
            std::vector<double> tempBiases = layers[i].biases;
            for (int j = 0; j < layers[i].biases.size(); j++)
            {

                layers[i].biases[j] += increment;
                for (int k = 0; k < inputs.size(); k++)
                {
                    wynikiSieci.push_back(forwardPass(inputs[k]));
                }
                layerGradient.push_back((srStrata(wynikiSieci, expectedValues) - ogStrata) / increment);
                layers[i].biases = constBiases;
                wynikiSieci.clear();
            }
            gradient.push_back(layerGradient);
            layerGradient.clear();
        }
        return gradient;
    }
    std::vector<std::vector<std::vector<double>>> weightNumGrad(double increment, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expectedValues)
    {
        std::vector<std::vector<double>> wynikiSieci;
        for (int i = 0; i < inputs.size(); i++)
        {
            wynikiSieci.push_back(forwardPass(inputs[i]));
        }
        double ogStrata = srStrata(wynikiSieci, expectedValues);
        wynikiSieci.clear();
        std::vector<std::vector<double>> layerGradient;
        std::vector<double> neuronGradient;
        std::vector<std::vector<std::vector<double>>> gradient;

        for (int i = 0; i < layers.size(); i++)
        {
            const std::vector<std::vector<double>> constWeights = layers[i].weights;
            std::vector<std::vector<double>> tempWeights = layers[i].weights;
            for (int j = 0; j < layers[i].weights.size(); j++)
            {
                for (int k = 0; k < layers[i].weights[j].size(); k++)
                {
                    layers[i].weights[j][k] += increment;
                    for (int l = 0; l < inputs.size(); l++)
                    {
                        wynikiSieci.push_back(forwardPass(inputs[l]));
                    }
                    neuronGradient.push_back((srStrata(wynikiSieci, expectedValues) - ogStrata) / increment);
                    layers[i].weights[j][k] = constWeights[j][k];
                    wynikiSieci.clear();
                }

                layerGradient.push_back(neuronGradient);
                neuronGradient.clear();
            }
            gradient.push_back(layerGradient);
            layerGradient.clear();
        }
        return gradient;
    }
    void update(std::vector<std::vector<std::vector<double>>> weightNumGrad, std::vector<std::vector<double>> biasNumGrad, double learningRate)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            for (int j = 0; j < layers[i].weights.size(); j++)
            {
                for (int k = 0; k < layers[i].weights[j].size(); k++)
                {
                    layers[i].weights[j][k] = layers[i].weights[j][k] - learningRate * weightNumGrad[i][j][k];
                }
            }
        }
        for (int i = 0; i < layers.size(); i++)
        {
            for (int j = 0; j < layers[i].biases.size(); j++)
            {
                layers[i].biases[j] = layers[i].biases[j] - learningRate * biasNumGrad[i][j];
            }
        }
    }
    void train(std::vector<std::vector<double>> trainingData, std::vector<std::vector<double>> expectedValues, double increment, double learningRate, int iterations)
    {
        for (int i = 0; i < iterations; i++)
        {
            update(weightNumGrad(increment, trainingData, expectedValues), biasNumGrad(increment, trainingData, expectedValues), learningRate);
            if (i % 1000 == 0)
            {
                save("model.json", trainingData, expectedValues);
            }
        }
    }
    void save(std::string fileName, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expectedValues)
    {
        std::vector<std::vector<double>> wynikiSieci;
        for (int i = 0; i < inputs.size(); i++)
        {
            wynikiSieci.push_back(forwardPass(inputs[i]));
        }

        double ogStrata = srStrata(wynikiSieci, expectedValues);
        json j;
        j["loss"] = ogStrata;
        j["layers"] = json::array();
        for (auto &layer : layers)
        {
            j["layers"].push_back({{"weights", layer.weights}, {"biases", layer.biases}});
        }
        std::ofstream file(fileName);
        file << j.dump(4);
        file.close();
    }
    void load(std::string fileName)
    {
        json j = json::array();
        std::ifstream file(fileName);
        file >> j;
        for (int i = 0; i < layers.size(); i++)
        {
            layers[i].weights = j["layers"][i]["weights"].get<std::vector<std::vector<double>>>();
            layers[i].biases = j["layers"][i]["biases"].get<std::vector<double>>(); 
        }

        file.close();
    }
    siec(std::vector<Warstwa> layers);
};
siec::siec(std::vector<Warstwa> layers) : layers(layers) {}

int main()
{
    std::vector<std::vector<double>> trainingData = {{1, 1}, {0, 0}, {1, 0}, {0, 1}}, expectedValues = {{0}, {0}, {1}, {1}};
    siec XOR({Warstwa(2, 2), Warstwa(2, 2), Warstwa(1, 2)});
    XOR.train(trainingData, expectedValues, 0.0001, 1, 100000);
}
