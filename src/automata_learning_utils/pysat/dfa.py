#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, io

import automata_learning_utils.dfa

class DFA(automata_learning_utils.dfa.DFA):
    """Deterministic Finite Automaton."""

    def __init__(self, *,
                 alphabet = [],
                 states = [],
                 transitions = {},
                 init_states = [],
                 accepting_states = [],
    ):
        """
            :type alphabet:         list(<Letter>)
            :type state:            list(<State>)
            :type transitions:      dict(tuple(<State>,<Letter>):<State>)
            :type init_states:      list(<State>)
            :type accepting_states: list(<State>)
        """
        self.alphabet = alphabet
        self.states = states
        self.transitions = transitions
        self.init_states = init_states
        self.accepting_states = accepting_states

    @classmethod
    def from_RPNI(cls, RPNI_output_file_name):
        dfa = automata_learning_utils.dfa.DFA.__new__(cls)
        automata_learning_utils.dfa.DFA.__init__(dfa, RPNI_output_file_name)
        return dfa

    def export_as_RPNI_automaton(self, output_file=sys.stdout, *,
        keep_alphabet=False,
    ):
        if isinstance(output_file, str):
            context = output_file = open(output_file, "w")
        else:
            context = open(os.devnull,"w")
        with context:
            output_file.write('[general]\n')
            output_file.write('\tis dfa = true;\n')
            output_file.write('\talphabet size = {};\n'.format(max(self.alphabet)+1 if keep_alphabet else len(self.alphabet)))
            output_file.write('\tnumber of states = {};\n'.format(len(self.states)))
            output_file.write('[initial states]\n')
            output_file.write('\t{};\n'.format(
                ', '.join(str(self.states.index(init_state)) for init_state in self.init_states)
            ))
            output_file.write('[final states]\n')
            output_file.write('\t{};\n'.format(
                ', '.join(str(self.states.index(accepting_state)) for accepting_state in self.accepting_states)
            ))
            output_file.write('[transitions]\n')
            for (state_from,letter),state_to in self.transitions.items():
                output_file.write('\t{}, {}, {};\n'.format(
                    self.states.index(state_from),
                    letter if keep_alphabet else self.alphabet.index(letter),
                    self.states.index(state_to),
                ))

    def export_as_visualization_dot(self, output_file=sys.stdout, *,
        keep_states=False, keep_alphabet=False,
        group_separator=r'\n',
    ):
        """
            :param group_separator: if set, group transitions between same pair of states. Usually set to ';' or r'\n'.
            :type group_separator: str or None
        """
        if isinstance(output_file, str):
            context = output_file = open(output_file, "w")
        else:
            context = open(os.devnull,"w")
        with context:
            """
                inspiration from:
                src/automata_learning_utils/lib_RPNI/libalf/src/conjecture.cpp
                line 448: string finite_automaton::visualize()
            """

            # head
            output_file.write('digraph finite_automaton {\n')
            output_file.write('\tgraph[fontsize=8];\n')
            output_file.write('\trankdir=LR;\n')
            output_file.write('\tsize=8;\n\n')

            # mark final states
            header_written = False
            final_state_count = 0
            const_iterator = {}

            if not keep_states:
                # final states
                if (len(self.accepting_states) > 0):
                    output_file.write('\tnode [shape=doublecircle, style="", color=black];')
                    for q,state in enumerate(self.states):
                        if state not in self.accepting_states: continue
                        output_file.write(' q{}'.format(q))
                    output_file.write(';\n')
                # normal states
                if (len(self.accepting_states) < len(self.states)):
                    output_file.write('\tnode [shape=circle, style="", color=black];')
                    for q,state in enumerate(self.states):
                        if state in self.accepting_states: continue
                        output_file.write(' q{}'.format(q))
                    output_file.write(';\n')
            else:
                # states
                for q,state in enumerate(self.states):
                    shape = 'doublecircle' if state in self.accepting_states else 'circle'
                    output_file.write('\tnode [shape={}, label="{}", style="", color=black]; q{};\n'.format(
                        shape,
                        state,
                        q,
                    ))

            # non-visible states for arrows to initial states
            if (len(self.init_states) > 0):
                output_file.write('\tnode [shape=plaintext, label="", style=""];')
                for iq,init_state in enumerate(self.init_states):
                    output_file.write(' iq{}'.format(iq))
                output_file.write(';\n')

            # and arrows to mark initial states
            for iq,init_state in enumerate(self.init_states):
                output_file.write('\tiq{} -> q{} [color=blue];\n'.format(
                    iq,
                    self.states.index(init_state)
                ))

            # transitions
            if group_separator is None:
                for (state_from,letter),state_to in self.transitions.items():
                    output_file.write('\tq{} -> q{} [label="{}"];\n'.format(
                        self.states.index(state_from),
                        self.states.index(state_to),
                        letter if keep_alphabet else self.alphabet.index(letter),
                    ))
            else:
                grouped_transitions = {}
                for (state_from,letter),state_to in self.transitions.items():
                    grouped_transitions.setdefault((state_from,state_to), set())
                    grouped_transitions[(state_from,state_to)].add(letter)
                for (state_from,state_to),letters in grouped_transitions.items():
                    output_file.write('\tq{} -> q{} [label="{}"];\n'.format(
                        self.states.index(state_from),
                        self.states.index(state_to),
                        group_separator.join(
                            "{}".format(letter if keep_alphabet else self.alphabet.index(letter))
                            for letter in sorted(letters, key=lambda l: self.alphabet.index(l))
                        ),
                    ))

            # end
            output_file.write('}\n')

    def size(self):
        return len(self.states)

    def _states(self):
        return set(self.states)

    def _initial_states(self):
        return set(self.init_states)

    def _terminal_states(self):
        return set(self.accepting_states)

    def _next_states(self, states, letter):
        return set(self.transitions[(state,letter)] for state in states)

    def test_word(self, word):
        current_states = self._initial_states()
        for letter in word:
            current_states = self._next_states(current_states,letter)
        return len(current_states & self._terminal_states()) != 0

    def __str__(self, word):
        writer = io.StringIO()
        self.export_as_RPNI_automaton(writer)
        return writer.getvalue()
    def __contains__(self, word):
        return self.test_word(word)
