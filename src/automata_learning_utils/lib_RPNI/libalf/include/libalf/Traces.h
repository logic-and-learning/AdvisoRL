//
// Created by gavran on 26.06.19.
//

#ifndef LIBALF_TRACES_H
#define LIBALF_TRACES_H

#include <list>
#include <string>

using namespace std;


class Traces {
public:
    int alphabet_size;
    Traces(string traces_filename);
    list<list<int>> positive_examples;
    list<list<int>> negative_examples;
    friend ostream& operator<<(ostream&, const Traces&);

};


#endif //LIBALF_TRACES_H
