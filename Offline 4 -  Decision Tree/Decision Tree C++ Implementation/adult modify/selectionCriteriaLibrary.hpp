#ifndef SELECTION_CRITERIA_LIBRARY_HPP
#define SELECTION_CRITERIA_LIBRARY_HPP

#include <bits/stdc++.h>
using namespace std;

#include "datasetLibrary.hpp"
#include "attributeLibrary.hpp"

enum SelectionCriteria {
    InformationGain,
    InformationGainRatio,
    NormalizedWeightedInformationGain
};

double entropy(vector<string> labels) {
    unordered_map<string, int> labelCount;
    for (const auto& label : labels) {
        labelCount[label]++;
    }

    double entropyValue = 0.0;
    double total = labels.size();

    for (const auto& pair : labelCount) {
        double probability = pair.second / total;
        entropyValue -= probability * log2(probability);
    }

    return entropyValue;
}

double IG(Dataset &dataset, Attributes &attribute) {
    cout<<"IG"<<endl;
    double totalEntropy = entropy(dataset.labels);
    double weightedEntropy = 0.0;
    
    if (attribute.type == "categorical") {
        map<string, vector<string>> subsets;

        for (size_t i = 0; i < dataset.rows.size(); ++i) {
            string attributeValue = dataset.rows[i][attribute.index];
            subsets[attributeValue].push_back(dataset.labels[i]);
        }

        for (auto& pair : subsets) {
            double subsetEntropy = entropy(pair.second);
            weightedEntropy += (pair.second.size() / static_cast<double>(dataset.rows.size())) * subsetEntropy;
        }

        return totalEntropy - weightedEntropy;        
    }

    if (attribute.type == "numerical") {
        set<double> values;
        for (auto& row : dataset.rows) {
            values.insert(stod(row[attribute.index]));
        }
        vector<double> sortedValues(values.begin(), values.end());
        sort(sortedValues.begin(), sortedValues.end());

        double bestIG = 0.0;
        double bestThreshold = 0.0;

        for (size_t i = 1; i < sortedValues.size(); ++i) {
            double threshold = (sortedValues[i - 1] + sortedValues[i]) / 2.0;

            vector<string> leftLabels;
            vector<string> rightLabels;

            for (size_t j = 0; j < dataset.rows.size(); ++j) {
                if (stod(dataset.rows[j][attribute.index]) <= threshold) {
                    leftLabels.push_back(dataset.labels[j]);
                } else {
                    rightLabels.push_back(dataset.labels[j]);
                }
            }

            double currentIG = totalEntropy - 
                (leftLabels.size() / static_cast<double>(dataset.rows.size())) * entropy(leftLabels) - 
                (rightLabels.size() / static_cast<double>(dataset.rows.size())) * entropy(rightLabels);
            if (currentIG > bestIG) {
                bestIG = currentIG;
                bestThreshold = threshold;
            }
        }
        attribute.threshold = bestThreshold;

        for (auto& attr : dataset.attributes) { 
            if (attr == attribute) {
                attr.threshold = bestThreshold;
                break;
            }
        }
        return bestIG;
    }

    return 0.0; 
}

double IGR(Dataset &dataset, Attributes &attribute) {
    double ig = IG(dataset, attribute); 
    double intrinsicValue = 0.0;
    double total = dataset.rows.size();

    if (attribute.type == "categorical") {
        map<string, int> valueCount;
        for (auto& row : dataset.rows) {
            valueCount[row[attribute.index]]++;
        }
        for (auto& pair : valueCount) {
            double probability = pair.second / total;
            if (probability > 0)
                intrinsicValue -= probability * log2(probability);
        }
    } else if (attribute.type == "numerical") {
        double threshold = attribute.threshold;
        int leftCount = 0, rightCount = 0;
        for (auto& row : dataset.rows) {
            double value = stod(row[attribute.index]);
            if (value <= threshold)
                leftCount++;
            else
                rightCount++;
        }
        double leftProb = leftCount / total;
        double rightProb = rightCount / total;
        if (leftProb > 0)
            intrinsicValue -= leftProb * log2(leftProb);
        if (rightProb > 0)
            intrinsicValue -= rightProb * log2(rightProb);
    }

    return (intrinsicValue != 0) ? (ig / intrinsicValue) : 0.0;
}

double NWIG(Dataset &dataset, Attributes &attribute) {
    double ig = IG(dataset, attribute);  
    double n = dataset.rows.size();

    double k = 0; 

    if (attribute.type == "categorical") {
        set<string> uniqueVals;
        for (auto& row : dataset.rows)
            uniqueVals.insert(row[attribute.index]);
        k = uniqueVals.size();
    } else if (attribute.type == "numerical") {
        k = 2;
    }
    if (k == 0 || n == 0) return 0;
    return (ig / log2(k + 1)) * (1 - (k - 1) / n);
}

double selectionCriteria(Dataset &dataset, Attributes &attribute, int criterion) {
    switch (criterion) {
        case InformationGain:
            return IG(dataset, attribute);
        case InformationGainRatio:
            return IGR(dataset, attribute);
        case NormalizedWeightedInformationGain:
            return NWIG(dataset, attribute);
        default:
            return 0.0;
    }
}

Attributes findBestAttribute(Dataset &dataset, int criterion) {
    double bestValue = -1.0;
    Attributes bestAttribute;
    
    for (auto& attribute : dataset.attributes) {
        double value = selectionCriteria(dataset, attribute, criterion);
        if (value > bestValue) {
            bestValue = value;
            bestAttribute = attribute;
        }
    }
    
    return bestAttribute;
}

#endif // SELECTION_CRITERIA_LIBRARY_HPP