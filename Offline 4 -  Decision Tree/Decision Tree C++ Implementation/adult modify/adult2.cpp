#include "attributeLibrary.hpp"
#include "datasetLibrary.hpp"
#include "selectionCriteriaLibrary.hpp"
#include "dtLibrary.hpp"

#include <bits/stdc++.h>
using namespace std;

// Updated CSV loader: skips column index 2 (workclass_code)
void loadIrisCSV(const string& filename, Dataset& dataset) {
    ifstream file(filename);
    string line; 

    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> cells;
        string cell;

        while (getline(ss, cell, ',')) {
            cells.push_back(cell);
        }

        // Skip row if the number of cells is not expected (+1 for label, -1 for removed column)
        if (cells.size() != dataset.attributes.size() + 2) {
            continue; // Skip malformed rows
        }

        vector<string> rowData;
        for (size_t i = 0; i < cells.size() - 1; ++i) {
            if (i == 2) continue; // Skip workclass_code
            rowData.push_back(cells[i]);
        }

        string label = cells.back();
        dataset.rows.push_back(rowData);
        dataset.labels.push_back(label);
    }
}

void run(Dataset& dataset) {
    int times = 1;
    vector<int> maxDepths = {13};
    vector<SelectionCriteria> criteria = {
        InformationGain,
        InformationGainRatio,
        NormalizedWeightedInformationGain
    };
    ofstream outputFile("adult_imputed_results.csv");
    outputFile << "Max Depth,Selection Criteria,TrainTime(s),TreeSize,Accuracy(%)\n";

    for (int maxDepth : maxDepths) {
        for (SelectionCriteria criterion : criteria) {
            string criterionName = (criterion == InformationGain ? "IG" :
                                   (criterion == InformationGainRatio ? "IGR" : "NWIG"));
            cout << "==== Running for MaxDepth = " << maxDepth 
                 << ", Criterion = " << criterionName << endl;

            double avgTrainTime = 0.0, avgAccuracy = 0.0, avgTreeSize = 0.0;

            pair<Dataset, Dataset> split = trainTestSplitRandom(dataset, 0.8);

            auto start = chrono::high_resolution_clock::now();
            DecisionTree dt(split.first, criterion, maxDepth);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> trainTime = end - start;
            avgTrainTime += trainTime.count();

            avgTreeSize += dt.getSize();

            auto testStart = chrono::high_resolution_clock::now();
            int correctPredictions = 0;
            for (auto& row : split.second.rows) {
                string predictedLabel = dt.predictLabel(row);
                string actualLabel = split.second.labels[&row - &split.second.rows[0]];
                if (predictedLabel == actualLabel) {
                    correctPredictions++;
                }
            }
            auto testEnd = chrono::high_resolution_clock::now();
            chrono::duration<double> testTime = testEnd - testStart;
            cout << " done in " << testTime.count() << "s" << endl;

            double accuracy = static_cast<double>(correctPredictions) / split.second.rows.size() * 100.0;
            avgAccuracy += accuracy;

            outputFile << maxDepth << ","
                       << criterionName << ","
                       << fixed << setprecision(4) << (avgTrainTime / times) << ","
                       << static_cast<int>(avgTreeSize / times) << ","
                       << fixed << setprecision(2) << (avgAccuracy / times) << endl;

            cout << "==== Finished MaxDepth = " << maxDepth 
                 << ", Criterion = " << criterionName << " ====\n"
                 << "  Avg Train Time: " << fixed << setprecision(4) << (avgTrainTime / times) << "s\n"
                 << "  Avg Tree Size: " << static_cast<int>(avgTreeSize / times) << "\n"
                 << "  Avg Accuracy: " << fixed << setprecision(2) << (avgAccuracy / times) << "%\n\n";
        }
    }

    outputFile.close();
}

int main(int argc, char* argv[]) {
    string criterionStr = (argc > 1) ? argv[1] : "IGR";
    int maxDepth = (argc > 2) ? stoi(argv[2]) : 4;

    SelectionCriteria criterion;
    if (criterionStr == "IG") {
        criterion = InformationGain;
    } else if (criterionStr == "IGR") {
        criterion = InformationGainRatio;
    } else if (criterionStr == "NWIG") {
        criterion = NormalizedWeightedInformationGain;
    } else {
        cout << "Invalid criterion. Choose IG, IGR, or NWIG.\n";
        return 1;
    }

    Dataset dataset;
    dataset.name = "Adult Dataset";

    // "workclass_code" removed from attribute list
    vector<Attributes> attributes = {
        Attributes("Age", "numerical", {}, 0),
        Attributes("workclass", "categorical", {"Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"}, 1),
        Attributes("education", "categorical", {"10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool", "Prof-school", "Some-college"}, 2),
        Attributes("education_num", "numerical", {}, 3),
        Attributes("marital-status", "categorical", {"Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"}, 4),
        Attributes("occupation", "categorical", {"Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"}, 5),
        Attributes("relationship", "categorical", {"Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"}, 6),
        Attributes("race", "categorical", {"Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"}, 7),
        Attributes("sex", "categorical", {"Female", "Male"}, 8),
        Attributes("capital-gain", "numerical", {}, 9),
        Attributes("capital-loss", "numerical", {}, 10),
        Attributes("hours-per-week", "numerical", {}, 11),
        Attributes("native-country", "categorical", {"Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"}, 12),
    };
    dataset.attributes = attributes;

    cout << "Loading dataset..." << endl;

    auto load_start = chrono::high_resolution_clock::now();
    loadIrisCSV("adult_imputed.data", dataset);
    auto load_end = chrono::high_resolution_clock::now();
    double loading_time = chrono::duration<double>(load_end - load_start).count();
    cout << "Loaded " << dataset.rows.size() << " rows." << endl;
    cout << "Loading time: " << loading_time << " s" << endl << endl;

    run(dataset);

    return 0;
}
