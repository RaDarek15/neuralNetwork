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

//-------------------CLASSES-------------------

class Warstwa
{
private:
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
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
    ~Warstwa();
};
Warstwa::Warstwa(int neuronCount, int inputCount)
{
    weights.resize(neuronCount);
    for (auto &i : weights)
        weights.resize(inputCount);
    biases.resize(neuronCount);
    std::mt19937 mt(time(nullptr));
    std::uniform_int_distribution<std::mt19937::result_type> dist6(-1, 1);
    for (int i = 0; i < neuronCount; i++)
    {
        for (int j = 0; j < inputCount; j++)
        {
            weights[i][j] = mt();
        }
    }
    for (int i = 0; i < neuronCount; i++)
    {
        biases[i] = mt();
    }
}
Warstwa::Warstwa(std::vector<std::vector<double>> weights, std::vector<double> biases) : biases(biases), weights(weights) {};

Warstwa::~Warstwa()
{
}

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
            layers[i].forwardPass(layerOutput);
        }
        return layerOutput;
    }
    double strata(double wynikSieci, double oczekiwany) { return pow((oczekiwany - wynikSieci), 2); };
    double srStrata(std::vector<std::vector<std::vector<double>>> wagi, std::vector<std::vector<double>> biasy)
    {
        
        //return (strata(XOR({0.0, 0.0}, wagi, biasy)[0], 0) + strata(XOR({1.0, 0.0}, wagi, biasy)[0], 1) + strata(XOR({0.0, 1.0}, wagi, biasy)[0], 1) + strata(XOR({1.0, 1.0}, wagi, biasy)[0], 0)) / 4;
    }
    siec(std::vector<Warstwa> layers);
    siec();
};
siec::siec(std::vector<Warstwa> layers) : layers(layers) {}
siec::~siec()
{
}

//-------------------GLOBAL FUNCTIONS-------------------

double strata(double wynikSieci, double oczekiwany) { return pow((oczekiwany - wynikSieci), 2); };

std::vector<double> warstwa(
    std::vector<double> inputy,
    std::vector<std::vector<double>> wagi,
    std::vector<double> biasy)
{
    std::vector<double> wyniki;

    for (int i = 0; i < wagi.size(); i++)
    {
        double wynik = 0;

        for (int j = 0; j < inputy.size(); j++)
        {
            wynik += inputy[j] * wagi[i][j];
        }

        wynik += biasy[i];

        wyniki.push_back(sigmoid(wynik));
    }

    return wyniki;
}

std::vector<double> XOR(
    std::vector<double> inputy,
    std::vector<std::vector<std::vector<double>>> wagi,
    std::vector<std::vector<double>> biasy)
{
    return warstwa(
        warstwa(
            warstwa(inputy, wagi[0], biasy[0]),
            wagi[1],
            biasy[1]),
        wagi[2],
        biasy[2]);
}
double srStrata(std::vector<std::vector<std::vector<double>>> wagi, std::vector<std::vector<double>> biasy)
{
    return (strata(XOR({0.0, 0.0}, wagi, biasy)[0], 0) + strata(XOR({1.0, 0.0}, wagi, biasy)[0], 1) + strata(XOR({0.0, 1.0}, wagi, biasy)[0], 1) + strata(XOR({1.0, 1.0}, wagi, biasy)[0], 0)) / 4;
}
std::vector<std::vector<std::vector<double>>> WeigtNumGrad(std::vector<std::vector<std::vector<double>>> wagi, std::vector<std::vector<double>> biasy, double increment)
{
    double ogStrata = srStrata(wagi, biasy);
    std::vector<std::vector<std::vector<double>>> changes = wagi;
    std::vector<std::vector<std::vector<double>>> tempWagi = wagi;
    for (int i = 0; i < wagi.size(); i++)
    {
        for (int j = 0; j < wagi[i].size(); j++)
        {
            for (int k = 0; k < wagi[i][j].size(); k++)
            {
                tempWagi[i][j][k] += increment;
                changes[i][j][k] = (srStrata(tempWagi, biasy) - ogStrata) / increment;
                tempWagi = wagi;
            }
        }
    }
    return changes;
}
std::vector<std::vector<double>> biasNumGrad(std::vector<std::vector<std::vector<double>>> wagi, std::vector<std::vector<double>> biasy, double increment)
{
    double ogStrata = srStrata(wagi, biasy);
    std::vector<std::vector<double>> changes = biasy;
    std::vector<std::vector<double>> tempBiasy = biasy;
    for (int i = 0; i < wagi.size(); i++)
    {
        for (int j = 0; j < wagi[i].size(); j++)
        {
            tempBiasy[i][j] += increment;
            changes[i][j] = (srStrata(wagi, tempBiasy) - ogStrata) / increment;
            tempBiasy = biasy;
        }
    }
    return changes;
}

void weightUpdate(std::vector<std::vector<std::vector<double>>> weightNumGrad, std::vector<std::vector<double>> biasNumGrad, std::vector<std::vector<std::vector<double>>> &wagi, std::vector<std::vector<double>> &biasy, double learningRate)
{
    for (int i = 0; i < wagi.size(); i++)
    {
        for (int j = 0; j < wagi[i].size(); j++)
        {
            for (int k = 0; k < wagi[i][j].size(); k++)
            {
                wagi[i][j][k] = wagi[i][j][k] - learningRate * weightNumGrad[i][j][k];
            }
        }
    }
    for (int i = 0; i < biasy.size(); i++)
    {
        for (int j = 0; j < biasy[i].size(); j++)
        {
            biasy[i][j] = biasy[i][j] - learningRate * biasNumGrad[i][j];
        }
    }
}
void saveModel(const std::vector<std::vector<std::vector<double>>> &wagi, const std::vector<std::vector<double>> &biasy, std::string fileName)
{
    json j;
    j["srStrata"] = srStrata(wagi, biasy);

    j["weights"] = wagi;
    j["biases"] = biasy;
    std::ofstream file(fileName);
    file << j.dump(4);
    file.close();
}
void loadModel(std::vector<std::vector<std::vector<double>>> &wagi, std::vector<std::vector<double>> &biasy, std::string fileName)
{
    std::ifstream file("model.json");
    json j;
    file >> j;
    wagi = j["weights"].get<std::vector<std::vector<std::vector<double>>>>();
    biasy = j["biases"].get<std::vector<std::vector<double>>>();
}
int main()
{
    std::vector<std::vector<std::vector<double>>> wagi =
        {
            {{-0.5984, 0.9076},
             {-0.2183, -0.8423}},

            {{0.9145, 0.6585},
             {-0.7523, -0.0304}},

            {{-0.0834, 0.5}}};

    std::vector<std::vector<double>> biasy =
        {
            {0.3926, -0.8681},
            {0.8218, 0.2938},
            {0.1}};

    std::vector<double> inputy = {1.0, 0.0};

    loadModel(wagi, biasy, "model.json");
    std::vector<double> wyniki = XOR(inputy, wagi, biasy);
    for (int i = 0; i < 100000; i++)
    {
        weightUpdate(WeigtNumGrad(wagi, biasy, 0.0001), biasNumGrad(wagi, biasy, 0.0001), wagi, biasy, 1);
        if (i % 1000 == 0)
        {
            std::cout << srStrata(wagi, biasy) << std::endl;
            saveModel(wagi, biasy, "model.json");
        }
    }
}