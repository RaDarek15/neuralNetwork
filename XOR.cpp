#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include "json.hpp"

using json = nlohmann::json;

double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}
double deriSigmoid(double z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

struct layerCache
{
    std::vector<double> preActivation;
    std::vector<double> postActivation;
};
class Warstwa
{
private:
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    layerCache forwardPass(
        std::vector<double> inputy)
    {
        layerCache cache;

        for (int i = 0; i < weights.size(); i++)
        {
            double wynik = 0;

            for (int j = 0; j < inputy.size(); j++)
            {
                wynik += inputy[j] * weights[i][j];
            }

            wynik += biases[i];
            cache.preActivation.push_back(wynik);
            cache.postActivation.push_back(sigmoid(wynik));
        }

        return cache;
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
std::vector<double> error(std::vector<double> preActivation, std::vector<double> outputs, std::vector<double> expectedValues)
{
    std::vector<double> error;
    for (int i = 0; i < preActivation.size(); i++)
    {
        error.push_back(2 * (outputs[i] - expectedValues[i]) * deriSigmoid(preActivation[i]));
    }
    return error;
}
std::vector<double> propagateError(std::vector<double> error, Warstwa &layer, std::vector<double> preActivation)
{
    std::vector<double> out(layer.weights[0].size());

    for (int k = 0; k < layer.weights[0].size(); k++)
    {
        for (int j = 0; j < layer.weights.size(); j++)
        {
            out[k] += error[j] * layer.weights[j][k];
        }
        out[k] *= deriSigmoid(preActivation[k]);
    }
    return out;
}
class siec
{
private:
public:
    std::vector<Warstwa> layers;

    std::vector<layerCache> forwardPass(std::vector<double> inputs)
    {
        std::vector<layerCache> caches;
        std::vector<double> current = inputs;

        for (int i = 0; i < layers.size(); i++)
        {
            layerCache cache = layers[i].forwardPass(current);
            caches.push_back(cache);
            current = cache.postActivation;
        }
        return caches;
    }
    double srStrata(std::vector<std::vector<double>> wynikSieci, std::vector<std::vector<double>> expectedValues)
    {
        double srStrata = 0.0;
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

    std::vector<std::vector<std::vector<double>>> weightBackprop(std::vector<double> error, std::vector<layerCache> cache, std::vector<double> inputs)
    {
        std::vector<std::vector<std::vector<double>>> output(layers.size());
        for (int i = 0; i < layers.size(); i++)
        {
            output[i].resize(layers[i].weights.size());
            for (int j = 0; j < layers[i].weights.size(); j++)
            {
                output[i][j].resize(layers[i].weights[j].size());
            }
        }
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].weights.size(); j++)
            {
                for (int k = 0; k < layers[i].weights[j].size(); k++)
                {
                    if (i > 0)
                    {
                        output[i][j][k] = error[j] * cache[i - 1].postActivation[k];
                    }

                    else
                    {
                        output[i][j][k] = error[j] * inputs[k];
                    }
                }
            }
            if (i > 0)
            {
                error = propagateError(error, layers[i], cache[i - 1].preActivation);
            }
        }
        return output;
    }
    std::vector<std::vector<double>> biasBackprop(std::vector<double> error, std::vector<layerCache> cache, std::vector<double> inputs)
    {
        std::vector<std::vector<double>> output(layers.size());
        for (int i = 0; i < layers.size(); i++)
        {
            output[i].resize(layers[i].biases.size());
        }
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            for (int j = 0; j < layers[i].weights.size(); j++)
            {
                output[i][j] = error[j];
            }
            if (i > 0)
            {
                error = propagateError(error, layers[i], cache[i - 1].preActivation);
            }
        }
        return output;
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
                layers[i].biases[j] = layers[i].biases[j] - learningRate * biasNumGrad[i][j];
            }
        }
    }
    void train(std::vector<std::vector<double>> trainingData, std::vector<std::vector<double>> expectedValues, double learningRate, int iterations)
    {
        std::vector<layerCache> cache;
        for (int i = 0; i < iterations; i++)
        {

            for (int j = 0; j < trainingData.size(); j++)
            {
                cache = forwardPass(trainingData[j]);
                std::vector<double> err = error(cache.back().preActivation, cache.back().postActivation, expectedValues[j]);
                update(weightBackprop(err, cache, trainingData[j]), biasBackprop(err, cache, trainingData[j]), learningRate);
            }
            if (i % 1000 == 0)
            {
                std::vector<std::vector<double>> allOutputs;
                for (int j = 0; j < trainingData.size(); j++)
                    allOutputs.push_back(forwardPass(trainingData[j]).back().postActivation);


                std::cout << "iteration: " << i << " loss: " << srStrata(allOutputs, expectedValues) << "\n";
                save("model.json", trainingData, expectedValues);
            }
        }
    }

    void save(std::string fileName, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> expectedValues)
    {
        std::vector<std::vector<double>> wynikiSieci;
        for (int i = 0; i < inputs.size(); i++)
        {
            wynikiSieci.push_back(forwardPass(inputs[i]).back().postActivation);
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
    XOR.train(trainingData, expectedValues, 1, 100000);
}
