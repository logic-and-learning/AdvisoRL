import argparse
import pdb
import subprocess
import os, sys
import functools, itertools
from tester.timer import Timer
from automata_learning.Traces import Traces
from .dfa import DFA
from .pysat.dfa import DFA as Hint

al_timer = Timer()

def show_hint(hint, show_plots=True):
    """run xdot"""
    dirname = os.path.abspath(os.path.dirname(__file__))
    output_visualization_filename = os.path.join(dirname, "data/hint.dot")
    hint.export_as_visualization_dot(output_visualization_filename,
        keep_alphabet=True,
        group_separator=";",
    )
    if show_plots:
        subprocess.run(["xdot", output_visualization_filename])

def gen_hints(identifier, show_plots=False):
    if False:
        pass
    # elif identifier=="?":
    #     return [
    #         gen_sup_hint_dfa(symbols, show_plots=False)
    #         for symbols in itertools.product(Traces.letters)
    #     ]
    # else:
    #     return [gen_sup_hint_dfa(identifier, show_plots=show_plots)]
    else:
        char2dom = {
            **{letter: [letter] for letter in Traces.letters},
            '?': Traces.letters,
        }
        symbols_domain = [char2dom[s] for s in identifier]
        return [
            gen_sup_hint_dfa(symbols, show_plots=False)
            for symbols in itertools.product(*symbols_domain)
        ]

def gen_sup_hint_dfa(symbols, show_plots=False):
    """Generate a DFA that accepts *a*b*c* when symbols are "abc"."""
    hint = Hint(
        alphabet = Traces.numbers,
        states = list(range(len(symbols)+1)),
        init_states = [0],
        accepting_states = [len(symbols)],
    )
    hint.transitions = {
        (state,letter): state # self transition
        for state in hint.states
        for letter in hint.alphabet
    }
    for i,s in enumerate(symbols):
        a = Traces.letter2number(s.lower())
        hint.transitions[(i,a)] = i+1
    hint.name = ''.join(symbols)
    show_hint(hint, show_plots=show_plots)
    return hint

def gen_partial_hint_dfa(symbols, show_plots=False):
    """Same as `gen_sup_hint_dfa` but accepting state is no more a self loop (accept *a*b*c)."""
    hint = Hint(
        alphabet = Traces.numbers,
        states = list(range(len(symbols)+2)),
        init_states = [0],
        accepting_states = [len(symbols)],
    )
    hint.transitions = {
        (state,letter): state # self transition
        for state in hint.states
        for letter in hint.alphabet
    }
    for i,s in enumerate(symbols):
        a = Traces.letter2number(s.lower())
        hint.transitions[(i,a)] = i+1
    for s in all_symbols:
        a = Traces.letter2number(s.lower())
        hint.transitions[(len(symbols),a)] = len(symbols)+1
    show_hint(hint, show_plots=show_plots)
    return hint
def gen_empty_hint_dfa(symbols, show_plots=False):
    """Structural hint: desactivate accepting state."""
    hint = gen_sup_hint_dfa(symbols, show_plots=False)
    hint.accepting_states = []
    show_hint(hint, show_plots=show_plots)
    return hint

def gen_dfa_from_hints(sup_hints=[], sub_hints=[], pysat_algorithm=None, show_plots=False):
    dirname = os.path.abspath(os.path.dirname(__file__))
    output_filename = os.path.join(dirname, "data/automaton.txt")
    output_visualization_filename = os.path.join(dirname, "data/hypothesis.dot")

    from .pysat import pysat_data_file
    if pysat_algorithm is not None:
        pysat_data_file = functools.partial(pysat_data_file, sat_algorithm=pysat_algorithm)

    if (len(sup_hints)+len(sub_hints) == 1):
        dfa = list(itertools.chain(sup_hints,sub_hints))[0]
        dfa.export_as_visualization_dot(output_visualization_filename,
            keep_alphabet=True,
            group_separator=";",
        )
    else:
        # merge several hints
        # FIXME: alays produce empty dfa
        dfa = pysat_data_file(None, output_filename, output_visualization_filename ,sup_hints=sup_hints, sub_hints=sub_hints)

    if show_plots:
        subprocess.run(["xdot", output_visualization_filename])
    return dfa

def learn_automaton(traces_filename,
                    show_plots = False,
                    is_SAT = None, # depreciated
                    *,
                    automaton_learning_algorithm = None,
                    pysat_algorithm = None,
                    automaton_learning_program = None,
                    sup_hint_dfas = None,
                    output_reward_machine_filename = None,
):
    """
        You can specify either the Automaton learning algorithm or the program itself.

        :param show_plots: run xdot on the result automaton if True
        :param is_SAT: DEPRECIATED: use ``automaton_learning_algorithm``="SAT"/"RPNI" instead of True/False
        :param automaton_learning_algorithm: "SAT"|"RPNI"|"PYSAT"
        :param pysat_algorithm: "RC2"|"FM"|"GC3"
        :param automaton_learning_program: either a path to an exectuable or a function.
            Will be passed thoses arguments: ``traces_filename, output_filename, output_visualization_filename``.
        :param sup_hint_dfas: hints to use with PYSAT.
        :param output_reward_machine_filename: default "./data/rm.txt"
        :return: ``output_visualization_filename`` (default "./data/hypothesis.dot")

        :type traces_filename: str
        :type show_plots: bool
        :type is_SAT: bool
        :type automaton_learning_algorithm: None or str
        :type pysat_algorithm: None or str
        :type automaton_learning_program: None, str or callable
        :type sup_hint_dfas: None or list(DFA)
        :type output_reward_machine_filename: str
        :rtype: str
    """


    dirname = os.path.abspath(os.path.dirname(__file__))
    show_hints = 1

    if output_reward_machine_filename is None:
        output_reward_machine_filename = os.path.join(dirname, "data/rm.txt")

    # `is_SAT` is depreciated, this is for backward compatibility:
    if is_SAT is not None:
        automaton_learning_algorithm = "SAT" if is_SAT else "RPNI"

    from .pysat import pysat_data_file
    if pysat_algorithm is not None:
        pysat_data_file = functools.partial(pysat_data_file, sat_algorithm=pysat_algorithm)

    if automaton_learning_algorithm is not None:
        automaton_learning_algorithm = automaton_learning_algorithm.upper()
        if automaton_learning_algorithm == "SAT":
            automaton_learning_program = os.path.join(dirname, "lib_SAT/libalf/testsuites/SAT/sat_data_file")
        elif automaton_learning_algorithm == "RPNI":
            automaton_learning_program = os.path.join(dirname, "lib_RPNI/libalf/testsuites/RPNI/rpni_data_file")
        elif automaton_learning_algorithm == "PYSAT":
            if sup_hint_dfas is None:
                automaton_learning_program = pysat_data_file
            else:
                show_hint(sup_hint_dfas[0], show_plots=(show_plots and show_hints))
                automaton_learning_program = functools.partial(pysat_data_file, sup_hints=sup_hint_dfas)
        # elif "PYSAT:HINT" in automaton_learning_algorithm:
        #     hints_labels = automaton_learning_algorithm.lower().split(":")[2:]
        #     hints = [gen_sup_hint_dfa(symbols) for symbols in hints_labels]
        #     show_hint(hints[0], show_plots=(show_plots and show_hints))
        #     automaton_learning_program = functools.partial(pysat_data_file, sup_hints=hints)
        # elif automaton_learning_algorithm == "PYSAT:PSEUDOHINT":
        #     hint = Hint(
        #         alphabet = Traces.numbers,
        #         states = list(range(4)),
        #         init_states = [0],
        #     )
        #     hint.accepting_states = hint.states
        #     hint.transitions = {
        #         (state,letter): state # self transition
        #         for state in hint.states
        #         for letter in hint.alphabet
        #     }
        #     hint.transitions[(0,Traces.letter2number('e'))] = 1
        #     hint.transitions[(1,Traces.letter2number('g'))] = 2
        #     hint.transitions[(2,Traces.letter2number('c'))] = 3
        #     show_hint(hint, show_plots=(show_plots and show_hints))
        #     automaton_learning_program = functools.partial(pysat_data_file, sup_hints=[hint])
        else:
            raise ValueError(f"Unknown automaton learning algorithm: {automaton_learning_algorithm}")
    elif automaton_learning_program is None:
        raise RuntimeError("Automaton learning program hasn't been specified")


    output_filename = os.path.join(dirname, "data/automaton.txt")
    output_visualization_filename = os.path.join(dirname, "data/hypothesis.dot")
    with al_timer.include():
        if isinstance(automaton_learning_program, str):
            subprocess.run([automaton_learning_program, traces_filename, output_filename, output_visualization_filename])
        elif callable(automaton_learning_program):
            automaton_learning_program(traces_filename, output_filename, output_visualization_filename)
        else:
            raise TypeError("Unable to run automaton learning program")
    if show_plots:
        subprocess.run(["xdot", output_visualization_filename])

    dfa = DFA(output_filename)
    dfa.export_as_reward_automaton(output_reward_machine_filename)
    return os.path.abspath(output_visualization_filename)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_filename")

    args = parser.parse_args()
    learn_automaton(args.traces_filename)


if __name__ == '__main__':
    main()
