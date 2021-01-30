#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import itertools

from .dfa import DFA
from .problem import Problem

sat_algorithms = [
    "rc2",
    "fm",
    "gc3", # unweighted clauses only
]

def read_RPNI_samples(input_file):
    if isinstance(input_file, str):
        context = input_file = open(input_file, "r")
    else:
        context = open(os.devnull,"r")
    with context:
        lines = input_file.read().splitlines()

    def read_word(line):
        return [int(l.strip()) for l in line.split(",")]

    cursor = iter(lines)

    positive_sample = []
    negative_sample = [[]]
    for line in cursor:
        if "POSITIVE" in line:
            current_sample = positive_sample
        elif "NEGATIVE" in line:
            current_sample = negative_sample
        else:
            if not line: continue
            current_sample.append(read_word(line))

    alphabet = get_alphabet(itertools.chain.from_iterable(itertools.chain(positive_sample, negative_sample)))

    return (alphabet, positive_sample, negative_sample)


def extract_samples_from_traces(traces):
    positive_sample = list(traces.positive)
    negative_sample = list(traces.negative)
    alphabet = get_alphabet(itertools.chain.from_iterable(itertools.chain(positive_sample, negative_sample)))
    return (alphabet, positive_sample, negative_sample)


def get_alphabet(letters):
    # nv = max(letters)
    # alphabet = list(range(nv+1))

    ls = set(letters)
    alphabet = sorted(ls)

    return alphabet

def sat_data_file(traces=None, output_filename=None, output_visualization_filename=None, *,
                  sat_algorithm=sat_algorithms[0],
                  sup_hints=[], sub_hints=[],
                  weight_sample=None, weight_hint=None,
                  min_size=1, max_size=None,
):
    if traces is None:
        positive_sample, negative_sample = [], []
        alphabet = get_alphabet(itertools.chain.from_iterable(hint.alphabet for hint in itertools.chain(sup_hints, sub_hints)))
    elif isinstance(traces, str): # it's a file name
        alphabet, positive_sample, negative_sample = read_RPNI_samples(traces)
    elif isinstance(traces, tuple):
        alphabet, positive_sample, negative_sample = traces
    else: # it's a Trace object
        alphabet, positive_sample, negative_sample = extract_samples_from_traces(traces)

    assert len(alphabet) > 0

    oldrecursionlimit = sys.getrecursionlimit() # TODO: find a better way
    sys.setrecursionlimit(10000)
    try:

        # eliminate inconsistent hints
        sup_hints = [hint for hint in sup_hints if all(trace     in hint for trace in positive_sample)]
        sub_hints = [hint for hint in sub_hints if all(trace not in hint for trace in negative_sample)]
        print("\x1B[1;34m... hints are {}.\x1B[m".format(
            ", ".join(hint.name for hint in sup_hints),
        ))

        for N in itertools.count(min_size):

            if max_size is not None and N > max_size:
                raise RuntimeError("DFA with at most {} states not found".format(max_size))
                # return None

            print("\x1B[1m>>> trying to solve DFA with {} states.\x1B[m".format(N))

            problem = Problem(N, alphabet)

            problem.add_positive_traces(positive_sample, weight_sample)
            problem.add_negative_traces(negative_sample, weight_sample)
            problem.add_positive_hints(sub_hints, weight_hint)
            problem.add_negative_hints(sup_hints, weight_hint)
            # for n in range(N):
            #     problem.add_negative_traces(
            #         itertools.product(alphabet, repeat=N),
            #     weight=1)

            wcnf = problem.build_cnf()
            print("\x1B[1;34m... translated to {} CNF clauses of {} vars.\x1B[m".format(
                len(wcnf.hard)+len(wcnf.soft),
                wcnf.nv,
            ))

            success = problem.solve(sat_algorithm)
            if success:
                print("\x1B[1;34m... {}: satisfiable.\x1B[m".format(sat_algorithm))
                break

            print("\x1B[1;34m... {}: unsatisfiable.\x1B[m".format(sat_algorithm))

    finally:
        sys.setrecursionlimit(oldrecursionlimit) # TODO: find a better way

    dfa = problem.get_automaton()

    # for word in positive_sample:
    #     if not dfa.test_word(word): print("WARNING: ",word)
    # for word in negative_sample:
    #     if dfa.test_word(word): print("WARNING: ",word)

    if output_filename is not None:
        dfa.export_as_RPNI_automaton(output_filename,
            keep_alphabet=True,
        )
    if output_visualization_filename is not None:
        dfa.export_as_visualization_dot(output_visualization_filename,
            keep_alphabet=True,
            group_separator=";",
        )

    return dfa
