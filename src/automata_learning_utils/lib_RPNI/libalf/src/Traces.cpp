//
// Created by gavran on 26.06.19.
//

#include "../include/libalf/Traces.h"
#include <iostream>
#include <fstream>
#include <sstream>

Traces::Traces(string traces_filename) {
    string line;
    ifstream data_file(traces_filename);
    int max_symbol = 0;
    enum Mode {start, positive, negative};
    Mode m = start;
    if (data_file.is_open()){
        while (getline(data_file, line)){
            if (line.compare("POSITIVE:") == 0){

                m = positive;
            }
            else if (line.compare("NEGATIVE:") == 0){

                m = negative;
            }
            else {
                istringstream tokenStream(line);
                list<int> example;
                string w;
                while (getline(tokenStream, w, ',')) {

                    int symbol = stoi(w);
                    if (symbol > max_symbol) {
                        max_symbol = symbol;
                    }
                    example.push_back(symbol);
                }

                if (m == positive) {
                    positive_examples.push_back(example);
                } else if (m == negative) {
                    negative_examples.push_back(example);
                }
            }
        }
    }
    alphabet_size = max_symbol + 1;
}


ostream& operator<<(ostream &strm, const Traces &t){
    strm<<"Alphabet size: "<<t.alphabet_size<<endl;
    strm<<"positive: "<<endl;
    for (auto it = t.positive_examples.begin(); it != t.positive_examples.end(); ++it){
        for (auto innerIt = it->begin(); innerIt != it->end(); ++innerIt){
            strm<<*innerIt<<" ";
        }
        strm<<endl;
    }
    strm<<endl;

    strm<<"negative: "<<endl;
    for (auto it = t.negative_examples.begin(); it != t.negative_examples.end(); ++it){
        for (auto innerIt = it->begin(); innerIt != it->end(); ++innerIt){
            strm<<*innerIt<<" ";
        }
        strm<<endl;
    }

    return strm;
}