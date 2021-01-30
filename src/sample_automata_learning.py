from automata_learning_utils.al_utils import learn_automaton
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    dirname = os.path.abspath(os.path.dirname(__file__))


    default_traces = os.path.join(dirname, "automata_learning_utils/data/data_test.txt")

    parser.add_argument("--traces_filename", default=default_traces)
    args = parser.parse_args()

    learn_automaton(args.traces_filename)


if __name__ == '__main__':
    main()