# Automata Learning Utils
This part of the repo includes RPNI algorithm from [libalf](http://libalf.informatik.rwth-aachen.de/).
Libalf is written in C++. This folder includes python wrapper around it (at the moment,
the *wrapper* only means calling the right file using system call).

## Installation
Enter the folder `lib/libalf`. Write `sudo make`. 
This will install the files to /usr/local/lib (on Ubuntu). 

## Usage
An example usage from inside the `qrm/src` folder is given in the file 
[sample_automata_learning.py](../sample_automata_learning.py)

That scripts expects the file with traces (example is `src/automata_learning_utils/data/data.txt`).
The learned automaton will be translated to a reward_machine and placed in `src/automata_learning_utils/data/data.txt`.
For the visualization purposes, one can view the automaton usint `xdot src/automata_learning_utils/data/hypothesis.dot`
