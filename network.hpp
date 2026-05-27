#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <omp.h>
#include <functional>
#include "json.hpp"

using json = nlohmann::json;

double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}
double deriSigmoid(double z)
{
    double s = sigmoid(z);
    return s * (1 - s);
}
double leakReLU(double z)
{
    if (z > 0)
    {
        return z;
    }
    return 0.01 * z;
}
double deriLeakReLU(double z)
{
    if (z > 0)
    {
        return 1;
    }
    return 0.01;
}
std::vector<double> softmax(const std::vector<double> &z)
{
    std::vector<double> out(z.size());
    double max = *std::max_element(z.begin(), z.end());
    double sum = 0;
    for (int i = 0; i < z.size(); i++)
    {
        out[i] = exp(z[i] - max);
        sum += out[i];
    }
    for (int i = 0; i < z.size(); i++)
        out[i] /= sum;
    return out;
}
double crossEntropy(const std::vector<double> &z, const std::vector<double> &expected)
{
    double L = 0;
    for (int i = 0; i < z.size(); i++)
    {
        L -= expected[i] * log(z[i] + 1e-9);
    }
    return L;
}
struct layerCache
{
    std::vector<double> preActivation;
    std::vector<double> postActivation;
};

class Warstwa
{
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
    layerCache forwardPass(const std::vector<double> &inputy)
    {
        layerCache cache;
        int neuronCount = weights.size();
        cache.preActivation.resize(neuronCount);
        cache.postActivation.resize(neuronCount);

        for (int i = 0; i < neuronCount; i++)
        {
            double wynik = biases[i];
            for (int j = 0; j < (int)inputy.size(); j++)
            {
                wynik += inputy[j] * weights[i][j];
            }
            cache.preActivation[i] = wynik;
            cache.postActivation[i] = activation(wynik);
        }
        return cache;
    }

    Warstwa(int neuronCount, int inputCount, std::function<double(double)> activation, std::function<double(double)> activationDerivative);
};

Warstwa::Warstwa(int neuronCount, int inputCount, std::function<double(double)> activation, std::function<double(double)> activationDerivative) : activation(activation), activationDerivative(activationDerivative)
{
    weights.resize(neuronCount, std::vector<double>(inputCount));
    biases.resize(neuronCount);
    std::mt19937 mt(time(nullptr));
    std::normal_distribution<double> dist(0, sqrt(2.0 / inputCount));
    for (int i = 0; i < neuronCount; i++)
        for (int j = 0; j < inputCount; j++)
            weights[i][j] = dist(mt);
    for (int i = 0; i < neuronCount; i++)
        biases[i] = dist(mt);
}

std::vector<double> computeError(const std::vector<double> &preActivation, const std::vector<double> &outputs, const std::vector<double> &expectedValues)
{
    std::vector<double> err(preActivation.size());
    for (int i = 0; i < (int)preActivation.size(); i++)
        err[i] = 2 * (outputs[i] - expectedValues[i]) * deriSigmoid(preActivation[i]);
    return err;
}

std::vector<double> propagateError(const std::vector<double> &err,
                                   const Warstwa &layer,
                                   const std::vector<double> &preActivation,
                                   std::function<double(double)> prevActivationDerivative)
{
    int inputSize = layer.weights[0].size();
    std::vector<double> out(inputSize, 0.0);
    for (int k = 0; k < inputSize; k++)
    {
        for (int j = 0; j < (int)layer.weights.size(); j++)
            out[k] += err[j] * layer.weights[j][k];
        out[k] *= prevActivationDerivative(preActivation[k]);
    }
    return out;
}

class siec
{
public:
    std::vector<Warstwa> layers;

    std::vector<layerCache> forwardPass(const std::vector<double> &inputs)
    {
        std::vector<layerCache> caches;
        caches.reserve(layers.size());
        std::vector<double> current = inputs;
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layerCache cache = layers[i].forwardPass(current);
            current = cache.postActivation;
            caches.push_back(std::move(cache));
        }
        return caches;
    }
    std::vector<std::vector<std::vector<double>>> weightBackprop(
        std::vector<double> err,
        const std::vector<layerCache> &cache,
        const std::vector<double> &inputs)
    {
        std::vector<std::vector<std::vector<double>>> output(layers.size());
        for (int i = 0; i < (int)layers.size(); i++)
        {
            output[i].resize(layers[i].weights.size());
            for (int j = 0; j < (int)layers[i].weights.size(); j++)
                output[i][j].resize(layers[i].weights[j].size());
        }
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            const std::vector<double> &prevActivation = (i > 0) ? cache[i - 1].postActivation : inputs;
            for (int j = 0; j < (int)layers[i].weights.size(); j++)
                for (int k = 0; k < (int)layers[i].weights[j].size(); k++)
                    output[i][j][k] = err[j] * prevActivation[k];
            if (i > 0)
                err = propagateError(err, layers[i], cache[i - 1].preActivation, layers[i - 1].activationDerivative);
        }
        return output;
    }

    std::vector<std::vector<double>> biasBackprop(
        std::vector<double> err,
        const std::vector<layerCache> &cache,
        const std::vector<double> &inputs)
    {
        std::vector<std::vector<double>> output(layers.size());
        for (int i = 0; i < (int)layers.size(); i++)
            output[i].resize(layers[i].biases.size());
        for (int i = layers.size() - 1; i >= 0; i--)
        {
            for (int j = 0; j < (int)layers[i].weights.size(); j++)
                output[i][j] = err[j];
            if (i > 0)
                err = propagateError(err, layers[i], cache[i - 1].preActivation, layers[i - 1].activationDerivative);
        }
        return output;
    }

    void update(const std::vector<std::vector<std::vector<double>>> &weightGrad,
                const std::vector<std::vector<double>> &biasGrad,
                double learningRate)
    {
        for (int i = 0; i < (int)layers.size(); i++)
            for (int j = 0; j < (int)layers[i].weights.size(); j++)
            {
                for (int k = 0; k < (int)layers[i].weights[j].size(); k++)
                    layers[i].weights[j][k] -= learningRate * weightGrad[i][j][k];
                layers[i].biases[j] -= learningRate * biasGrad[i][j];
            }
    }

    void train(const std::vector<std::vector<double>> &trainingData,
               const std::vector<std::vector<double>> &expectedValues,
               double learningRate,
               int iterations,
               int batchSize = 32)
    {
        int n = trainingData.size();

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(10);

        for (int epoch = 0; epoch < iterations; epoch++)
        {
            std::shuffle(indices.begin(), indices.end(), rng);

            for (int batchStart = 0; batchStart < n; batchStart += batchSize)
            {
                int batchEnd = std::min(batchStart + batchSize, n);
                int actualBatchSize = batchEnd - batchStart;
                std::vector<std::vector<std::vector<double>>> accumWeightGrad(layers.size());
                std::vector<std::vector<double>> accumBiasGrad(layers.size());
                for (int i = 0; i < (int)layers.size(); i++)
                {
                    accumWeightGrad[i].resize(layers[i].weights.size());
                    for (int j = 0; j < (int)layers[i].weights.size(); j++)
                        accumWeightGrad[i][j].assign(layers[i].weights[j].size(), 0.0);
                    accumBiasGrad[i].assign(layers[i].biases.size(), 0.0);
                }
#pragma omp parallel
                {
                    std::vector<std::vector<std::vector<double>>> localWeightGrad(layers.size());
                    std::vector<std::vector<double>> localBiasGrad(layers.size());
                    for (int i = 0; i < (int)layers.size(); i++)
                    {
                        localWeightGrad[i].resize(layers[i].weights.size());
                        for (int j = 0; j < (int)layers[i].weights.size(); j++)
                            localWeightGrad[i][j].assign(layers[i].weights[j].size(), 0.0);
                        localBiasGrad[i].assign(layers[i].biases.size(), 0.0);
                    }

#pragma omp for nowait
                    for (int b = batchStart; b < batchEnd; b++)
                    {
                        int idx = indices[b];
                        auto cache = forwardPass(trainingData[idx]);
                        std::vector<double> err(expectedValues[idx].size());
                        auto s = softmax(cache.back().postActivation);
                        for (int i = 0; i < err.size(); i++)
                        {
                            err[i] = s[i] - expectedValues[idx][i];
                        }

                        auto wGrad = weightBackprop(err, cache, trainingData[idx]);
                        auto bGrad = biasBackprop(err, cache, trainingData[idx]);

                        for (int i = 0; i < (int)layers.size(); i++)
                        {
                            for (int j = 0; j < (int)layers[i].weights.size(); j++)
                            {
                                for (int k = 0; k < (int)layers[i].weights[j].size(); k++)
                                    localWeightGrad[i][j][k] += wGrad[i][j][k];
                                localBiasGrad[i][j] += bGrad[i][j];
                            }
                        }
                    }
#pragma omp critical
                    {
                        for (int i = 0; i < (int)layers.size(); i++)
                            for (int j = 0; j < (int)layers[i].weights.size(); j++)
                            {
                                for (int k = 0; k < (int)layers[i].weights[j].size(); k++)
                                    accumWeightGrad[i][j][k] += localWeightGrad[i][j][k];
                                accumBiasGrad[i][j] += localBiasGrad[i][j];
                            }
                    }
                }

                double scale = learningRate / actualBatchSize;
                for (int i = 0; i < (int)layers.size(); i++)
                    for (int j = 0; j < (int)layers[i].weights.size(); j++)
                    {
                        for (int k = 0; k < (int)layers[i].weights[j].size(); k++)
                            accumWeightGrad[i][j][k] *= scale;
                        accumBiasGrad[i][j] *= scale;
                    }
                update(accumWeightGrad, accumBiasGrad, 1.0);
            }

            std::cout << "Epoch " << epoch + 1 << "/" << iterations << " done\n";
            save("model.json");
        }
    }

    double srStrata(const std::vector<std::vector<double>> &wynikSieci,
                    const std::vector<std::vector<double>> &expectedValues)
    {
        double total = 0.0;
        for (int i = 0; i < (int)expectedValues.size(); i++)
            for (int j = 0; j < (int)expectedValues[i].size(); j++)
                total += pow(expectedValues[i][j] - wynikSieci[i][j], 2);
        return total / expectedValues.size();
    }

    void save(const std::string &fileName)
    {
        json j;
        j["layers"] = json::array();
        for (auto &layer : layers)
            j["layers"].push_back({{"weights", layer.weights}, {"biases", layer.biases}});
        std::ofstream file(fileName);
        file << j.dump(4);
    }

    void load(const std::string &fileName)
    {
        json j;
        std::ifstream file(fileName);
        file >> j;
        for (int i = 0; i < (int)layers.size(); i++)
        {
            layers[i].weights = j["layers"][i]["weights"].get<std::vector<std::vector<double>>>();
            layers[i].biases = j["layers"][i]["biases"].get<std::vector<double>>();
        }
    }

    double accuracy(const std::vector<std::vector<double>> &wynikSieci,
                    const std::vector<std::vector<double>> &expectedValues)
    {
        int counter = 0;
        for (int i = 0; i < (int)expectedValues.size(); i++)
        {
            int predicted = std::max_element(wynikSieci[i].begin(), wynikSieci[i].end()) - wynikSieci[i].begin();
            int expected = std::max_element(expectedValues[i].begin(), expectedValues[i].end()) - expectedValues[i].begin();
            if (expected == predicted)
                counter++;
        }
        return (double)counter / expectedValues.size() * 100.0;
    }
    std::vector<double> loadPng(const std::string &filename)
    {

        int width, height, channels;
        std::vector<double> out(784);
        double src_x;
        double src_y;
        unsigned char *data = stbi_load(filename.c_str(), &width, &height, &channels, 1);
        std::vector<double> image(width * height);
        for (int i = 0; i < width * height; i++)
        {
            image[i] = 1.0 - data[i] / 255.0;
        }

        stbi_image_free(data);
        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                src_x = x * (width / 28.0);
                src_y = y * (height / 28.0);

                int x0 = std::floor(src_x);
                int x1 = std::ceil(src_x);
                int y0 = std::floor(src_y);
                int y1 = std::ceil(src_y);
                x1 = std::min(x1, width - 1);
                y1 = std::min(y1, height - 1);
                double fx = src_x - x0;
                double fy = src_y - y0;
                out[y * 28 + x] = image[y0 * width + x0] * (1 - fx) * (1 - fy) + image[y1 * width + x0] * (fx * (1 - fy)) + image[y0 * width + x1] * ((1 - fx) * fy) + image[y1 * width + x1] * (fy * fx);
            }
        }
        return out;
    }
    siec(const std::vector<Warstwa> &layers) : layers(layers) {}
};
