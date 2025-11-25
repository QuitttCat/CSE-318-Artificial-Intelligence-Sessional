#ifndef DATASET_LIBRARY_HPP
#define DATASET_LIBRARY_HPP

#include "attributeLibrary.hpp"
#include <bits/stdc++.h>
using namespace std;

class Dataset {
public:
    string name;
    vector<Attributes> attributes;
    vector<vector<string>> rows;
    vector<string> labels;

    Dataset(string name, vector<Attributes> &attributes, vector<vector<string>> &rows, vector<string> &labels) 
        : name(name), attributes(attributes), rows(rows), labels(labels) {
        for (size_t i = 0; i < attributes.size(); ++i) {
            this->attributes[i].index = i;
        }
    }
    Dataset() {}
    
    string getMajorityLabel() {
        unordered_map<string, int> labelCount;
        for (const auto& label : labels) {
            labelCount[label]++;
        }
        string majorityLabel;
        int maxCount = 0;
        for (const auto& pair : labelCount) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                majorityLabel = pair.first;
            }
        }
        return majorityLabel;
    }
};

pair<Dataset, Dataset> trainTestSplitRandom(Dataset &dataset, double trainSize) {
    vector<vector<string>> trainRows;
    vector<vector<string>> testRows;
    vector<string> trainLabels;
    vector<string> testLabels;

    random_device rd;
    mt19937 g(rd());
    
    // Create indices for shuffling
    vector<size_t> indices(dataset.rows.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), g);

    size_t trainCount = static_cast<size_t>(dataset.rows.size() * trainSize);
    
    for (size_t i = 0; i < dataset.rows.size(); ++i) {
        if (i < trainCount) {
            trainRows.push_back(dataset.rows[indices[i]]);
            trainLabels.push_back(dataset.labels[indices[i]]);
        } else {
            testRows.push_back(dataset.rows[indices[i]]);
            testLabels.push_back(dataset.labels[indices[i]]);
        }
    }
    return make_pair(
        Dataset(dataset.name, dataset.attributes, trainRows, trainLabels),
        Dataset(dataset.name, dataset.attributes, testRows, testLabels)
    );
}

Dataset filterByCategorical(const Dataset& dataset, const Attributes& attr, const string& value) {
    vector<Attributes> newAttributes;
    size_t attrIndex = attr.index; 
    for (const auto& a : dataset.attributes) {
        if (a.name != attr.name) {
            newAttributes.push_back(a);
        }
    }
    
    vector<vector<string>> newRows;
    vector<string> newLabels;

    for (size_t i = 0; i < dataset.rows.size(); ++i) {
        if (dataset.rows[i][attrIndex] == value) {
            vector<string> newRow;
            for (size_t j = 0; j < dataset.rows[i].size(); ++j) {
                if (j != attrIndex) {
                    newRow.push_back(dataset.rows[i][j]);
                }
            }
            newRows.push_back(newRow);
            newLabels.push_back(dataset.labels[i]);
        }
    }
    
    for (size_t i = 0; i < newAttributes.size(); ++i) {
        newAttributes[i].index = i;
    }
    return Dataset(dataset.name, newAttributes, newRows, newLabels);
}

Dataset filterByNumerical(const Dataset& dataset, const Attributes& attr, double threshold, bool lessEqual) {
    vector<Attributes> newAttributes;
    size_t attrIndex = attr.index; 
    for (const auto& a : dataset.attributes) {
        if (a.name != attr.name) {
            newAttributes.push_back(a);
        }
    }
    
    vector<vector<string>> newRows;
    vector<string> newLabels;

    for (size_t i = 0; i < dataset.rows.size(); ++i) {
        double val = stod(dataset.rows[i][attrIndex]);
        if ((lessEqual && val <= threshold) || (!lessEqual && val > threshold)) {
            vector<string> newRow;
            for (size_t j = 0; j < dataset.rows[i].size(); ++j) {
                if (j != attrIndex) {
                    newRow.push_back(dataset.rows[i][j]);
                }
            }
            newRows.push_back(newRow);
            newLabels.push_back(dataset.labels[i]);
        }
    }
    
    for (size_t i = 0; i < newAttributes.size(); ++i) {
        newAttributes[i].index = i;
    }
    return Dataset(dataset.name, newAttributes, newRows, newLabels);
}

void printDataset(Dataset& dataset) {
    cout << "Dataset: " << dataset.name << endl;
    cout << "Attributes:" << endl;
    for (auto& attr : dataset.attributes) {
        cout << "  " << attr << ", Index: " << attr.index << endl;
    }
    cout << "Rows:" << endl;
    for (size_t i = 0; i < dataset.rows.size(); ++i) {
        cout << "  ";
        for (size_t j = 0; j < dataset.rows[i].size() - 1; ++j) {
            cout << dataset.attributes[j].name << ": " << dataset.rows[i][j] << ", ";
        }
        cout << "Label: " << dataset.labels[i] << endl;
    }
}

#endif // DATASET_LIBRARY_HPP