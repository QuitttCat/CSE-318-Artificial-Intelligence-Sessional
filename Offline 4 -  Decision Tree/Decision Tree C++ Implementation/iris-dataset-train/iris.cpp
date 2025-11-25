#include "attributeLibrary.hpp"
#include "datasetLibrary.hpp"
#include "selectionCriteriaLibrary.hpp"
#include "DTLibrary.hpp"

#include <bits/stdc++.h>
using namespace std;


void loadIrisCSV(const string& filename, Dataset& dataset) {
    ifstream file(filename);
    string line;
    getline(file, line); 

    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> cells;
        string cell;

        while (getline(ss, cell, ',')) {
            cells.push_back(cell);
        }

        map<Attributes, string> data;

        for (size_t i = 0; i < dataset.attributes.size(); ++i) {
            data[dataset.attributes[i]] = cells[i + 1]; 
        }
        string label = cells.back(); 

        dataset.rows.emplace_back(data, label);
        dataset.labels.push_back(label);
    }
}

void run(Dataset& dataset) {
    vector<int> maxDepths = {1, 2, 3, 4, 5,6};
    vector<SelectionCriteria> criteria = {InformationGain, InformationGainRatio, NormalizedWeightedInformationGain};
    ofstream outputFile("iris_results.csv");
    outputFile << "Max Depth,Selection Criteria,trainTime(ms),treeSize,Accuracy(%)\n";

    for (int maxDepth : maxDepths) {
        for (SelectionCriteria criterion : criteria) {
            double avgTrainTime = 0.0, avgAccuracy = 0.0, avgTreeSize = 0.0;
            for (int i = 1; i <= 20; i++) {
                pair<Dataset, Dataset> split = trainTestSplitRandom(dataset, 0.8);
                auto start = chrono::high_resolution_clock::now();
                DecisionTree dt(split.first, criterion, maxDepth);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> trainTime = end - start;
                avgTrainTime += trainTime.count();
                avgTreeSize += dt.getSize();

                int correctPredictions = 0;
                for (auto& row : split.second.rows) {
                    string predictedLabel = dt.predictLabel(row);
                    if (predictedLabel == row.label) {
                        correctPredictions++;
                    }
                }

                avgAccuracy += static_cast<double>(correctPredictions) / split.second.rows.size() * 100;
            }

            outputFile << maxDepth << "," << criterion << "," << avgTrainTime / 20 << "," << (int)(avgTreeSize / 20) << "," << avgAccuracy / 20 << "\n";
        }
    }
}

int main()
{
    Dataset dataset;
    dataset.name = "Iris Dataset";
    vector<Attributes> attributes = {  
        Attributes("sepal_length", "numerical", {}),
        Attributes("sepal_width", "numerical", {}),
        Attributes("petal_length", "numerical", {}),
        Attributes("petal_width", "numerical", {})
    };
    dataset.attributes = attributes;
    loadIrisCSV("iris.csv", dataset);
    //printDataset(dataset);


    // cout << left << setw(25) << "Attribute" << setw(25) << "IG" << setw(25) << "IGR" << setw(25) << "NWIG" << endl;
    // for (auto& attr : dataset.attributes) {
    //     double ig = IG(dataset, attr);
    //     double igr = IGR(dataset, attr);
    //     double nwig = NWIG(dataset, attr);
    //     cout << left << setw(25) << attr.name << setw(25) << ig << setw(25) << igr << setw(25) << nwig << endl;
    // }

    run(dataset);

    return 0;
}