#include <iostream>
#include <vector>
#include <fstream>
#include <random>

struct data
{
    std::vector<std::vector<double>> images;
    std::vector<std::vector<double>> labels;
};

std::vector<int> readHeader(std::string filename)
{
    std::vector<int> out(4);
    std::vector<unsigned char> current(4);
    std::ifstream file(filename, std::ios::binary);

    for (int i = 0; i < 4; i++)
    {
        file.read((char *)current.data(), 4);
        out[i] = (current[0] << 24) | (current[1] << 16) | (current[2] << 8) | current[3];
    }

    return out;
}
std::vector<std::vector<double>> images(std::string filename, std::vector<int> header)
{
    unsigned char current;
    std::vector<std::vector<double>> output(header[1]);
    std::ifstream file(filename, std::ios::binary);
    file.seekg(16);
    for (int i = 0; i < header[1]; i++)
    {
        output[i].resize(header[2] * header[3]);
        for (int j = 0; j < header[2] * header[3]; j++)
        {
            file.read((char *)&current, 1);
            output[i][j] = current / 255.0;
        }
    }
    return output;
}
std::vector<std::vector<double>> labels(std::string filename, std::vector<int> header)
{
    unsigned char current;
    std::vector<std::vector<double>> output(header[1]);
    std::ifstream file(filename, std::ios::binary);
    file.seekg(8);
    for (int i = 0; i < header[1]; i++)
    {
        output[i].resize(10);
        file.read((char *)&current, 1);
        output[i][current] = 1.0;
    }
    return output;
}
data readMnist(std::string imagefileName,std::string labelfileName)
{
    data output;
    std::vector<int> imageHeader = readHeader(imagefileName);
    std::vector<int> labelHeader = readHeader(labelfileName);
    output.images = images(imagefileName,imageHeader);
    output.labels = labels(labelfileName,labelHeader);
    return output;
}