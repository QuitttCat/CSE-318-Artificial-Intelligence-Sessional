#include "attributeLibrary.hpp"
#include "datasetLibrary.hpp"
#include "selectionCriteriaLibrary.hpp"
#include "DTLibrary.hpp"

#include <bits/stdc++.h>
using namespace std;


void loadIrisCSV(const string& filename, Dataset& dataset) {
    ifstream file(filename);
    string line; 

    while (getline(file, line)) {
        //cout<<line<<endl;
        stringstream ss(line);
        vector<string> cells;
        string cell;

        while (getline(ss, cell, ',')) {
            cells.push_back(cell);
        }

        map<Attributes, string> data;

        for (size_t i = 0; i < dataset.attributes.size(); ++i) {
            data[dataset.attributes[i]] = cells[i]; 
            
        }
        string label = cells.back(); 

        dataset.rows.emplace_back(data, label);
        dataset.labels.push_back(label);
    }
}

void run(Dataset& dataset) {
    vector<int> maxDepths = {1, 2, 3, 4, 5,6};
    vector<SelectionCriteria> criteria = {InformationGain, InformationGainRatio, NormalizedWeightedInformationGain};
    ofstream outputFile("adult_imputed.data");
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

// int main()
// {
//     Dataset dataset;
//     dataset.name = "Adult Dataset";
// vector<Attributes> attributes = {
//     Attributes("Age", "numerical", {}),
//     Attributes("workclass", "categorical", {"Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"}),
//     Attributes("workclass_code", "numerical", {}),
//     Attributes("education", "categorical", {"10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool", "Prof-school", "Some-college"}),
//     Attributes("education_num", "numerical", {}),
//     Attributes("marital-status", "categorical", {"Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"}),
//     Attributes("occupation", "categorical", {"Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"}),
//     Attributes("relationship", "categorical", {"Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"}),
//     Attributes("race", "categorical", {"Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"}),
//     Attributes("sex", "categorical", {"Female", "Male"}),
//     Attributes("capital-gain", "numerical", {}),
//     Attributes("capital-loss", "numerical", {}),
//     Attributes("hours-per-week", "numerical", {}),
//     Attributes("native-country", "categorical", {"Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"}),
// };

//     dataset.attributes = attributes;
//     loadIrisCSV("adult_imputed.data", dataset);
//     //printDataset(dataset);


//     cout << left << setw(25) << "Attribute" << setw(25) << "IG" << setw(25) << "IGR" << setw(25) << "NWIG" << endl;
//     for (auto& attr : dataset.attributes) {
//         double ig = IG(dataset, attr);
//         cout<<ig<<endl;
//     }

//     //run(dataset);

//     return 0;
// }



int main(int argc, char* argv[])
{
    

    // string criterionStr = argv[1];
    // int maxDepth = stoi(argv[2]);

    string criterionStr = "IGR";
    int maxDepth = 5;

    // Map string to your enum or type
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
    vector<Attributes> attributes = {
        Attributes("Age", "numerical", {}),
        Attributes("workclass", "categorical", {"Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"}),
        Attributes("workclass_code", "numerical", {}),
        Attributes("education", "categorical", {"10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool", "Prof-school", "Some-college"}),
        Attributes("education_num", "numerical", {}),
        Attributes("marital-status", "categorical", {"Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"}),
        Attributes("occupation", "categorical", {"Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"}),
        Attributes("relationship", "categorical", {"Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"}),
        Attributes("race", "categorical", {"Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"}),
        Attributes("sex", "categorical", {"Female", "Male"}),
        Attributes("capital-gain", "numerical", {}),
        Attributes("capital-loss", "numerical", {}),
        Attributes("hours-per-week", "numerical", {}),
        Attributes("native-country", "categorical", {"Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"}),
    };
    dataset.attributes = attributes;


    
 cout << "Loading dataset..." << endl;
    auto load_start = std::chrono::high_resolution_clock::now();
    loadIrisCSV("adult_imputed.data", dataset);
    auto load_end = std::chrono::high_resolution_clock::now();
    double loading_time = std::chrono::duration<double>(load_end - load_start).count();
    cout << "Loaded " << dataset.rows.size() << " rows." << endl;
    cout << "Loading time: " << loading_time << " ms" << endl << endl;

    // Train/Test Split
    cout << "Splitting train/test data..." << endl;
    auto split_start = std::chrono::high_resolution_clock::now();
    pair<Dataset, Dataset> split = trainTestSplitRandom(dataset, 0.8);
    auto split_end = std::chrono::high_resolution_clock::now();
    double split_time = std::chrono::duration<double>(split_end - split_start).count();
    cout << "Training set size: " << split.first.rows.size() << ", Test set size: " << split.second.rows.size() << endl;
    cout << "Split time: " << split_time << " s" << endl << endl;

    // Training
    cout << "Training decision tree..." << endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    DecisionTree dt(split.first, criterion, maxDepth);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_time = std::chrono::duration<double>(train_end - train_start).count();
    cout << "Training completed." << endl;
    cout << "Training time: " << train_time << " s" << endl << endl;

    // Prediction
    cout << "Predicting test data..." << endl;
    auto pred_start = std::chrono::high_resolution_clock::now();
    int correctPredictions = 0;
    for (auto& row : split.second.rows) {
        string predictedLabel = dt.predictLabel(row);
        if (predictedLabel == row.label) {
            correctPredictions++;
        }
    }
    auto pred_end = std::chrono::high_resolution_clock::now();
    double prediction_time = std::chrono::duration<double>(pred_end - pred_start).count();

    double accuracy = static_cast<double>(correctPredictions) / split.second.rows.size() * 100.0;
    cout << "Prediction completed." << endl;
    cout << "Prediction time: " << prediction_time << " s" << endl << endl;

    // Output summary
    cout << "Summary:\n";
    cout << "Criterion   : " << criterionStr << endl;
    cout << "Max Depth   : " << maxDepth << endl;
    cout << "Train Time  : " << train_time << " s" << endl;
    cout << "Predict Time: " << prediction_time << " s" << endl;
    cout << "Tree Size   : " << dt.getSize() << endl;
    cout << "Accuracy    : " << fixed << setprecision(2) << accuracy << " %" << endl;

    return 0;
}