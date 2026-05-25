#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include "json.hpp"
#include "mnist.hpp"

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
        const std::vector<double>& inputy)
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
    Warstwa(const std::vector<std::vector<double>>& weights, const std::vector<double>& biases);
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
Warstwa::Warstwa(const std::vector<std::vector<double>>& weights, const std::vector<double>& biases) : biases(biases), weights(weights) {};
std::vector<double> error(const std::vector<double>& preActivation, const std::vector<double>& outputs, const std::vector<double>& expectedValues)
{
    std::vector<double> error;
    for (int i = 0; i < preActivation.size(); i++)
    {
        error.push_back(2 * (outputs[i] - expectedValues[i]) * deriSigmoid(preActivation[i]));
    }
    return error;
}
std::vector<double> propagateError(const std::vector<double>& error, const Warstwa &layer, const std::vector<double>& preActivation)
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

    std::vector<layerCache> forwardPass(const std::vector<double>& inputs)
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
    double srStrata(const std::vector<std::vector<double>>& wynikSieci, const std::vector<std::vector<double>>& expectedValues)
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

    std::vector<std::vector<std::vector<double>>> weightBackprop(std::vector<double> error, const std::vector<layerCache>& cache, const std::vector<double>& inputs)
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
    std::vector<std::vector<double>> biasBackprop(std::vector<double> error, const std::vector<layerCache>& cache, const std::vector<double>& inputs)
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
    void update(const std::vector<std::vector<std::vector<double>>>& weightNumGrad, const std::vector<std::vector<double>>& biasNumGrad, double learningRate)
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
    void train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& expectedValues, double learningRate, int iterations)
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
            std::vector<std::vector<double>> allOutputs;
            for (int j = 0; j < trainingData.size(); j++)
                allOutputs.push_back(forwardPass(trainingData[j]).back().postActivation);
            std::cout << "iteration: " << i << " loss: " << srStrata(allOutputs, expectedValues) << "\n";
            save("model.json", expectedValues, allOutputs);
        }
    }

    void save(const std::string& fileName, const std::vector<std::vector<double>>& expectedValues, const std::vector<std::vector<double>>& alloutputs)
    {
        double ogStrata = srStrata(alloutputs, expectedValues);
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
    void load(const std::string& fileName)
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
    double accuracy(const std::vector<std::vector<double>>& wynikSieci, const std::vector<std::vector<double>>& expectedValues)
    {
        int counter = 0;
        int i;
        for (i = 0; i < expectedValues.size(); i++)
        {
            int predicted = std::max_element(wynikSieci[i].begin(), wynikSieci[i].end()) - wynikSieci[i].begin();
            int expceted = std::max_element(expectedValues[i].begin(), expectedValues[i].end()) - expectedValues[i].begin();
            if (expceted == predicted) 
            {
                counter++;
            }
            
        }
        return (double)counter/i*100;
        
    }
    siec(const std::vector<Warstwa>& layers);
};
siec::siec(const std::vector<Warstwa>& layers) : layers(layers) {}

int main()
{
    char yesNo;
    std::vector<std::vector<double>> wyniki;
    std::ifstream f("model.json");
    data data = readMnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    std::vector<std::vector<double>> trainingData = data.images, expectedValues = data.labels;
    siec mnist({Warstwa(128, 784), Warstwa(64, 128), Warstwa(10, 64)});
    if (f)
    {
        mnist.load("model.json");
        std::cout << "Model loaded! \n";
    }

    std::cout << "do you want to train more? (y/n)\n";
    std::cin >> yesNo;
    if (yesNo == 'y')
    {
        mnist.train(trainingData, expectedValues, 0.01, 3);
    }
    data = readMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    for (int i = 0; i < data.images.size() ; i++)
    {
        wyniki.push_back(mnist.forwardPass(data.images[i]).back().postActivation);
    }
    std::cout << "Accuracy: " << mnist.accuracy(wyniki, data.labels) << "%\n";
}