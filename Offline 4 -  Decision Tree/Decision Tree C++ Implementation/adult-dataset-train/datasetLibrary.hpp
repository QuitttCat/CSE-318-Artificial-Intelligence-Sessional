#ifndef DATASET_LIBRARY_HPP
#define DATASET_LIBRARY_HPP





#include "attributeLibrary.hpp"
#include<bits/stdc++.h>
using namespace std;


class Datarow {
public:
    map<Attributes,string> data;
    string label;

    Datarow(map<Attributes,string> &data, string label) 
        : data(data), label(label) {}

    Datarow() {}

};

class Dataset {
public:
    string name;
    vector<Attributes> attributes;
    vector<Datarow> rows;
    vector<string> labels;

    Dataset(string name, vector<Attributes> &attributes, vector<Datarow> &rows, vector<string> &labels) 
        : name(name), attributes(attributes), rows(rows), labels(labels) {}
    Dataset() {}
    string getMajorityLabel() {
        map<string, int> labelCount;
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
    vector<Datarow> trainRows;
    vector<Datarow> testRows;
    vector<string> trainLabels;
    vector<string> testLabels;

    random_device rd;
    mt19937 g(rd());
    shuffle(dataset.rows.begin(), dataset.rows.end(), g);

    size_t trainCount = static_cast<size_t>(dataset.rows.size() * trainSize);
    
    for (size_t i = 0; i < dataset.rows.size(); ++i) {
        if (i < trainCount) {
            trainRows.push_back(dataset.rows[i]);
            trainLabels.push_back(dataset.labels[i]);
        } else {
            testRows.push_back(dataset.rows[i]);
            testLabels.push_back(dataset.labels[i]);
        }
    }
    return make_pair(
        Dataset(dataset.name, dataset.attributes, trainRows, trainLabels),
        Dataset(dataset.name, dataset.attributes, testRows, testLabels)
    );
}



Dataset filterByCategorical(const Dataset& dataset, const Attributes& attr, const string& value) {
    vector<Attributes> newAttributes;
    for (const auto& a : dataset.attributes) {
        if (a.name != attr.name) newAttributes.push_back(a); 
    }
    vector<Datarow> newRows;
    vector<string> newLabels;

    for (const auto& row : dataset.rows) {
        if (row.data.at(attr) == value) {
            map<Attributes, string> newData = row.data;
            newData.erase(attr);
            newRows.emplace_back(newData, row.label);
            newLabels.push_back(row.label);
        }
    }
    return Dataset(dataset.name, newAttributes, newRows, newLabels);
}


Dataset filterByNumerical(const Dataset& dataset, const Attributes& attr, double threshold, bool lessEqual) {
    vector<Attributes> newAttributes;
    for (const auto& a : dataset.attributes) {
        if (a.name != attr.name) newAttributes.push_back(a); 
    }
    vector<Datarow> newRows;
    vector<string> newLabels;

    for (const auto& row : dataset.rows) {
        double val = stod(row.data.at(attr));
        if ((lessEqual && val <= threshold) || (!lessEqual && val > threshold)) {
            map<Attributes, string> newData = row.data;
            newData.erase(attr);
            newRows.emplace_back(newData, row.label);
            newLabels.push_back(row.label);
        }
    }
    return Dataset(dataset.name, newAttributes, newRows, newLabels);
}


void printDataset(Dataset& dataset) {
    cout << "Dataset: " << dataset.name << endl;
    cout << "Attributes:" << endl;
    for (auto& attr : dataset.attributes) {
        cout << "  " << attr << endl;
    }
    cout << "Rows:" << endl;
    for (auto& row : dataset.rows) {
        cout << "  ";
        for (auto& pair : row.data) {
            cout << pair.first.name << ": " << pair.second << ", ";
        }
        cout << "Label: " << row.label << endl;
    }
}


#endif // DATASET_LIBRARY_HPP
