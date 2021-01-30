//
// Created by gavran on 26.06.19.
//

#include <iostream>
#include <ostream>
#include <iterator>
#include <fstream>
#include <algorithm>

#include "libalf/alf.h"
#include "libalf/Traces.h"

//#include <libalf/algorithm_RPNI.h>
#include <libalf/algorithm_biermann_minisat.h>
#include <string>

using namespace std;
using namespace libalf;

int main(int argc, char**argv)
{
    ostream_logger log(&cout, LOGGER_DEBUG);

    knowledgebase<bool> knowledge;
    cout<<"sth";

    ofstream file;
    char filename[128];

    string output_automaton_filename;
    string output_automaton_dot_filename;



    if(argc < 2) {
        cout << "give the name of the file containing traces.\n";
        return -1;
    }

    if (argc < 3){
        output_automaton_filename = "automaton.txt";
    }
    else{
        output_automaton_filename = argv[2];
    }

    if (argc < 4){
        output_automaton_dot_filename = "hypothesis.dot";
    } else{
        output_automaton_dot_filename = argv[3];
        cout<<"setting file name to be "<<output_automaton_dot_filename<<endl;
    }

    // create sample set in knowledgebase

    Traces example_traces = Traces(argv[1]);
    cout<<example_traces<<endl;

    for (auto exampleIt = example_traces.positive_examples.begin(); exampleIt != example_traces.positive_examples.end(); ++exampleIt){
        knowledge.add_knowledge(*exampleIt, true);
    }


    for (auto exampleIt = example_traces.negative_examples.begin(); exampleIt != example_traces.negative_examples.end(); ++exampleIt){
        knowledge.add_knowledge(*exampleIt, false);
    }



    //RPNI<bool> rumps(&knowledge, &log, example_traces.alphabet_size);
    MiniSat_biermann<bool> diebels(&knowledge, &log, example_traces.alphabet_size);
    conjecture *cj;

    if(!diebels.conjecture_ready()) {
            log(LOGGER_WARN, "biermann says that no conjecture is ready! trying anyway...\n");
    }

    if( NULL == (cj = diebels.advance()) ) {
            log(LOGGER_ERROR, "advance() returned false!\n");
    } else {
        //snprintf(filename, 128, "hypothesis.dot");
        file.open(output_automaton_dot_filename);

        file << cj->visualize();

        file.close();
        printf("\n\nhypothesis saved.\n\n");

        file.open(output_automaton_filename);
        file << cj->write();
        file.close();

    }

    delete cj;
    return 0;
}

