#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "mnist.hpp"
#include "network.hpp"

int main()
{
    std::cout << "Using " << omp_get_max_threads() << " threads\n";

    char yesNo;
    std::vector<std::vector<double>> wyniki;
    std::ifstream f("model.json");

    data trainData = readMnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

    siec mnist({Warstwa(128, 784, leakReLU, deriLeakReLU), Warstwa(64, 128, leakReLU, deriLeakReLU), Warstwa(10, 64, [](double x)
                                                                                                             { return x; }, [](double x)
                                                                                                             { return 1; })});

    if (f)
    {
        mnist.load("model.json");
        std::cout << "Model loaded!\n";
    }

    std::cout << "Do you want to train more? (y/n)\n";
    std::cin >> yesNo;
    if (yesNo == 'y')
    {
        mnist.train(trainData.images, trainData.labels, 0.01, 3, 32);
    }

    data testData = readMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    wyniki.reserve(testData.images.size());
    for (int i = 0; i < (int)testData.images.size(); i++)
        wyniki.push_back(mnist.forwardPass(testData.images[i]).back().postActivation);

    std::cout << "Accuracy: " << mnist.accuracy(wyniki, testData.labels) << "%\n";

    std::cout << "Do you want to put in your own number? (y/n)\n";
    std::cin >> yesNo;
    if (yesNo == 'y')
    {
        std::string filePath;
        std::cout << "Put in the filepath\n";
        std::cin >> filePath;
        std::vector<double> inputVector = mnist.loadPng(filePath);
        std::cout << "\n--- Processed Input (28x28) ---\n";
        printMnistAscii(inputVector);
        std::cout << "-------------------------------------\n\n";

        auto results = mnist.forwardPass(inputVector);

        auto finalOutputs = results.back().postActivation;
        auto probabilities = softmax(finalOutputs);

        int predictedDigit = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
        std::cout << "Predicted Digit: " << predictedDigit << "\n";
        std::cout << "Confidence: " << probabilities[predictedDigit] * 100.0 << "%\n";
    }
}