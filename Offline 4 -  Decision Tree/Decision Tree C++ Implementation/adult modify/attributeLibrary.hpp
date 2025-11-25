#ifndef ATTRIBUTE_LIBRARY_HPP
#define ATTRIBUTE_LIBRARY_HPP

#include <bits/stdc++.h>
using namespace std;

class Attributes {
public:
    string name;
    string type;
    vector<string> uniqueValues;
    double threshold;
    int index;

    Attributes(string name, string type, vector<string> uniqueValues, int index)
        : name(name), type(type), uniqueValues(uniqueValues), threshold(0), index(index) {}

    Attributes(const Attributes& other)
        : name(other.name), type(other.type), uniqueValues(other.uniqueValues), threshold(other.threshold), index(other.index) {}

    Attributes() : threshold(0), index(0) {}

    Attributes& operator=(const Attributes& other) {
        if (this != &other) {
            name = other.name;
            type = other.type;
            uniqueValues = other.uniqueValues;
            threshold = other.threshold;
            index = other.index;
        }
        return *this;
    }

    bool operator==(const Attributes& other) const {
        return name == other.name && type == other.type;
    }

    bool operator<(const Attributes& other) const {
        return name < other.name || (name == other.name && type < other.type);
    }
};

inline ostream& operator<<(ostream& os, const Attributes& attr) {
    os << "Name: " << attr.name << ", Type: " << attr.type;
    if (attr.type == "numerical") {
        os << ", Threshold: " << attr.threshold;
    } else {
        os << ", Unique Values: {";
        for (const auto& value : attr.uniqueValues) {
            os << value << " ";
        }
        os << "}";
    }
    return os;
}

#endif // ATTRIBUTE_LIBRARY_HPP