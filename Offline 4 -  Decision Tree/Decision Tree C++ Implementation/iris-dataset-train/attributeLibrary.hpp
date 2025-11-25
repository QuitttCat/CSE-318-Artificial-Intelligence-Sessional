#ifndef ATTRIBUTE_LIBRARY_HPP
#define ATTRIBUTE_LIBRARY_HPP

#include<bits/stdc++.h>
using namespace std;

class Attributes {
public:
    string name;
    string type;
    set<string> uniqueValues;
    double threshold; 
    Attributes(string name,string type, set<string> uniqueValues) 
        : name(name), type(type), uniqueValues(uniqueValues) , threshold(0) {}
    Attributes(const Attributes& other) : name(other.name), type(other.type), uniqueValues(other.uniqueValues), threshold(other.threshold) {}
    Attributes()
    {
        
    }
    Attributes& operator=(const Attributes& other) {
        if (this != &other) {
            name = other.name;
            type = other.type;
            uniqueValues = other.uniqueValues;
            threshold = other.threshold;
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

#endif 