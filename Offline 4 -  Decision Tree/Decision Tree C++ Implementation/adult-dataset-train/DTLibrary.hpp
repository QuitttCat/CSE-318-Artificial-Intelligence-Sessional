#ifndef DT_LIBRARY_HPP
#define DT_LIBRARY_HPP

#include <bits/stdc++.h>
using namespace std;
#include "datasetLibrary.hpp"
#include "selectionCriteriaLibrary.hpp"
#include "attributeLibrary.hpp"

class Node {
public:
    bool isLeaf;
    string label;
    Attributes attribute; 

    map<string, Node*> children;
    void addChild(string attr, Node* child) {
        children[attr] = child;
    }

    Node()
    {
        
    }


    int getDepth(){
        if (isLeaf) {
            return 1;
        } else {
            int maxDepth = -1;
            for (const auto& child : children) {
                maxDepth = max(maxDepth, child.second->getDepth());
            }
            return maxDepth + 1;
        }
    }

    int getSize() {
        if (isLeaf) {
            return 1;
        } else {
            int totalSize = 0;
            for (const auto& child : children) {
                totalSize += child.second->getSize();
            }
            return totalSize + 1; 
        }
    }
};


class DecisionTree {
public:
    Node* root;
    enum SelectionCriteria criterion;
    Dataset dataset;
    int maxDepth;

    DecisionTree(Dataset &dataset, enum SelectionCriteria criterion, int maxDepth = INT_MAX)
        : dataset(dataset), criterion(criterion), maxDepth(maxDepth) {
        root = new Node();
        root->isLeaf = false;
        buildTree(*root, dataset, 0);
    }


    void setRoot(Node* newRoot) {
        root = newRoot;
    }

    int getDepth() {
        if (root) {
            return root->getDepth();
        }
        return 0;
    }

    int getSize() {
        if (root) {
            return root->getSize();
        }
        return 0;
    }

    void buildTree(Node & node, Dataset &dataset,int depth)
    {   
        //printDataset(dataset);
        if (depth >= maxDepth || dataset.attributes.empty()) {
            node.isLeaf = true;
            //cout<<"Reached leaf with "<<dataset.rows.size()<<" samples"<<endl;
            node.label = dataset.getMajorityLabel();
            return;
        }

        Attributes bestAttribute = findBestAttribute(dataset, criterion);
        node.attribute = bestAttribute;
        node.isLeaf = false;

        if( bestAttribute.type == "categorical") {
            for (auto& value : bestAttribute.uniqueValues) {
                Dataset subset = filterByCategorical(dataset, bestAttribute, value);
                Node* child = new Node();
                node.addChild(value, child);
                buildTree(*child, subset, depth + 1);
            }
        }
        else if (bestAttribute.type == "numerical") {
            double threshold = bestAttribute.threshold;
            Dataset leftSubset = filterByNumerical(dataset, bestAttribute, threshold, true);
            Dataset rightSubset = filterByNumerical(dataset, bestAttribute, threshold, false);

            Node* leftChild = new Node();
            Node* rightChild = new Node();

            node.addChild("≤ " + to_string(threshold), leftChild);
            node.addChild("> " + to_string(threshold), rightChild);

            if (!leftSubset.rows.empty()) buildTree(*leftChild, leftSubset, depth + 1);
            else {
                leftChild->isLeaf = true;
                leftChild->label = dataset.getMajorityLabel();
            }
            if (!rightSubset.rows.empty()) buildTree(*rightChild, rightSubset, depth + 1);
            else {
                rightChild->isLeaf = true;
                rightChild->label = dataset.getMajorityLabel();
            }
        }



    }

    string predictLabel(Datarow &row) {
        Node* currentNode = root;
        while (!currentNode->isLeaf) {
            if (currentNode->attribute.type == "categorical") {
                string value = row.data[currentNode->attribute];
                if (currentNode->children.find(value) != currentNode->children.end()) {
                    currentNode = currentNode->children[value];
                } else {
                    return ""; 
                }
            } else if (currentNode->attribute.type == "numerical") {
                double value = stod(row.data[currentNode->attribute]);
                string key = (value <= currentNode->attribute.threshold) ? "≤ " + to_string(currentNode->attribute.threshold) : "> " + to_string(currentNode->attribute.threshold);
                if (currentNode->children.find(key) != currentNode->children.end()) {
                    currentNode = currentNode->children[key];
                } else {
                    return ""; 
                }
            }
        }
        return currentNode->label;
    }

    void printPrefix(Node* node, string prefix = "") {
        if (node->isLeaf) {
            cout << prefix << "Leaf: " << node->label << endl;
        } else {
            cout << prefix << "Node: " << node->attribute.name << endl;
            for (const auto& child : node->children) {
                printPrefix(child.second, prefix + "  ");
            }
        }
    }


};

#endif // DT_LIBRARY_HPP