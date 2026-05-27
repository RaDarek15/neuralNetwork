#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cctype>
#include <omp.h>
#include "mnist.hpp"
#include "network.hpp"
void printMnistAscii(const std::vector<double> &imageVec)
{
    if (imageVec.size() != 784)
    {
        std::cerr << "Error: Vector size is " << imageVec.size() << ", expected 784.\n";
        return;
    }
    const std::string ASCII_CHARS = " .:-=+*#%@";

    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            double val = imageVec[y * 28 + x];
            if (val < 0.0)
                val = 0.0;
            if (val > 1.0)
                val = 1.0;

            int charIndex = (int)(val * (ASCII_CHARS.length() - 1));

            std::cout << ASCII_CHARS[charIndex] << ASCII_CHARS[charIndex];
        }
        std::cout << "\n";
    }
}
void printHeader()
{
    std::cout << "\n╔══════════════════════════════╗\n";
    std::cout << "║       MNIST Classifier       ║\n";
    std::cout << "║  Threads: " << std::setw(2) << omp_get_max_threads() << "                 ║\n";
    std::cout << "╚══════════════════════════════╝\n\n";
}

void printMenu()
{
    std::cout << "  [1] Train model\n";
    std::cout << "  [2] Evaluate accuracy\n";
    std::cout << "  [3] Predict from image\n";
    std::cout << "  [4] Save model\n";
    std::cout << "  [0] Exit\n";
    std::cout << "\n> ";
}

char prompt(const std::string &msg)
{
    char c;
    std::cout << msg << " (y/n): ";
    std::cin >> c;
    return std::tolower(c);
}

void doTrain(siec &model)
{
    std::cout << "\n[Train]\n";
    double lr;
    int epochs, batchSize;
    std::cout << "  Learning rate [0.01]: ";
    std::cin >> lr;
    std::cout << "  Epochs        [3]:    ";
    std::cin >> epochs;
    std::cout << "  Batch size    [32]:   ";
    std::cin >> batchSize;

    std::cout << "\n  Loading training data...\n";
    data trainData = readMnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    std::cout << "  Starting training...\n\n";
    model.train(trainData.images, trainData.labels, lr, epochs, batchSize);
    std::cout << "\n  Done.\n";
}

void doEvaluate(siec &model)
{
    std::cout << "\n[Evaluate]\n  Loading test data...\n";
    data testData = readMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    std::vector<std::vector<double>> wyniki;
    wyniki.reserve(testData.images.size());
    for (const auto &img : testData.images)
        wyniki.push_back(model.forwardPass(img).back().postActivation);

    std::cout << "  Accuracy: " << model.accuracy(wyniki, testData.labels) << "%\n";
}

void doPredict(siec &model)
{
    std::cout << "\n[Predict]\n  File path: ";
    std::string filePath;
    std::cin >> filePath;

    std::vector<double> input = model.loadPng(filePath);
    std::cout << "\n--- Input (28x28) ---\n";
    printMnistAscii(input);
    std::cout << "---------------------\n\n";

    auto outputs = model.forwardPass(input).back().postActivation;
    auto probs = softmax(outputs);
    int digit = std::max_element(probs.begin(), probs.end()) - probs.begin();

    std::cout << "  Predicted : " << digit << "\n";
    std::cout << "  Confidence: " << std::fixed << std::setprecision(1)
              << probs[digit] * 100.0 << "%\n";

    std::cout << "\n  All scores:\n";
    for (int i = 0; i < 10; ++i)
    {
        int bar = std::clamp(static_cast<int>(probs[i] * 20), 0, 20);
        std::string filled, empty;
        for (int b = 0; b < bar; ++b)
            filled += "█";
        for (int b = 0; b < 20 - bar; ++b)
            empty += "░";

        std::cout << "  " << i << " │" << filled << empty << "│ "
                  << std::setw(5) << std::setprecision(1) << probs[i] * 100 << "%\n";
    }
}

int main()
{
    printHeader();

    siec model({Warstwa(128, 784, leakReLU, deriLeakReLU),
                Warstwa(64, 128, leakReLU, deriLeakReLU),
                Warstwa(10, 64, [](double x)
                        { return x; }, [](double x)
                        { return 1; })});

    if (std::ifstream("model.json"))
    {
        model.load("model.json");
        std::cout << "Loaded model.json\n";
    }
    else
    {
        std::cout << "No saved model found\n";
    }

    int choice = -1;
    while (choice != 0)
    {
        std::cout << "\n";
        printMenu();
        std::cin >> choice;

        switch (choice)
        {
        case 1:
            doTrain(model);
            break;
        case 2:
            doEvaluate(model);
            break;
        case 3:
            doPredict(model);
            break;
        case 4:
            model.save("model.json");
            std::cout << "Saved to model.json\n";
            break;
        case 0:
            if (prompt("  Save before exit?") == 'y')
            {
                model.save("model.json");
                std::cout << "Saved.\n";
            }
            break;
        default:
            std::cout << "Unknown option.\n";
            break;
        }
    }
}