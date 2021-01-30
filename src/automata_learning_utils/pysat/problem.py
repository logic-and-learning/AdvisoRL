#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import itertools, functools, operator
import pysat.formula, pysat.solvers
from .classification import Classification
from .dfa import DFA

class Problem:

    def __init__(self, N, alphabet):
        """
            If a weight is not set, it is considered to be hard.

            :param N: number of nodes of the final automaton.
            :type N: int

            :param alphabet: isn't it that obvious?
            :type alphabet: list(<Letter>)

            :Example:

            >>> problem = Problem(...)
            >>> problem.add_positive_traces(...)
            >>> problem.add_negative_hints(...)
            >>> problem.build_cnf()
            >>> success = problem.solve()
            >>> automaton = problem.get_automaton()
        """
        self.restart(N, alphabet)

        # for trace in positive_sample:
        #     self.generate_clauses_trace(trace, True, weight=weight_sample)
        # for trace in negative_sample:
        #     self.generate_clauses_trace(trace, False, weight=weight_sample)
        # for hint in sub_hints:
        #     self.generate_clauses_hint(hint, True, weight=weight_hint)
        # for hint in sup_hints:
        #     self.generate_clauses_hint(hint, False, weight=weight_hint)


    def restart(self, N=None, alphabet=None):
        if N is not None: self.N = int(N)
        if alphabet is not None: self.alphabet = alphabet

        self.vpool = pysat.formula.IDPool()
        self.all_clauses = []

        self.trans_ids = self._new_vars(self.N * len(self.alphabet) * self.N)
        self.trans_id = lambda p, a, q: self.trans_ids[((a) * self.N + p) * self.N + q]
        self.term_ids = self._new_vars(self.N)
        self.term_id = lambda q: self.term_ids[q]

        self.generate_clauses_automaton()
        self.classification = Classification(
            label = self.generate_clauses_word_init(),
        )

        self.wcnf = None




    def _new_var(self):
        """Create a fresh variable."""
        return self.vpool._next()

    def _new_vars(self, count):
        """Create `count` fresh variables."""
        if not count: return ()
        start = self.vpool._next()
        stop = start + count - 1
        self.vpool.top = stop
        return range(start, stop+1)

    def _add_clauses(self, *clauses):
        self.all_clauses.extend(clauses)
        # self.all_clauses.append(itertools.chain.from_iterable(clauses))

    def generate_clauses_automaton(self):
        """
        """

        # AUTOMATON DETERMINISM

        exist_transition = (
            ([
                +self.trans_id(p, a, q)
                for q in range(self.N)
            ], None)
            for p in range(self.N)
            for a,letter in enumerate(self.alphabet)
        )

        unique_transition = (
            ([
                -self.trans_id(p, a, q1),
                -self.trans_id(p, a, q2),
            ], None)
            for p in range(self.N)
            for a,letter in enumerate(self.alphabet)
            for q1 in range(self.N)
            for q2 in range(self.N)
            if q1 != q2
        )

        self._add_clauses(
            exist_transition,
            unique_transition,
        )

    def generate_clauses_word_new(self):
        """
            Generate clauses for a new word.

            :return: word reach state ids
            :rtype: function(q:int):int
        """
        u_reach_ids = self._new_vars(self.N)
        u_reach_id  = lambda q: u_reach_ids[q]

        exist_state = (
            ([
                +u_reach_id(q)
                for q in range(self.N)
            ], None),
        )

        unique_state = (
            ([
                -u_reach_id(q1),
                -u_reach_id(q2),
            ], None)
            for q1 in range(self.N)
            for q2 in range(self.N)
            if q1 != q2
        )

        self._add_clauses(
            exist_state,
            unique_state,
        )

        return u_reach_id

    def generate_clauses_word_init(self):
        """
            :return: empty word reach state ids
            :rtype: function(q:int):int
        """
        e_reach_id = self.generate_clauses_word_new()

        initial_state_clauses = (
            ([
                +e_reach_id(0),
            ], None),
        )

        self._add_clauses(
            initial_state_clauses,
        )

        return e_reach_id

    def generate_clauses_word_trans(self, u_reach_id, letter):
        """
            :param u_reach_id: prefix reach state ids
            :type: function(q:int):int

            :return: word reach state ids
            :rtype: function(q:int):int
        """
        a = self.alphabet.index(letter)
        ua_reach_id = self.generate_clauses_word_new()

        transition_state_clauses = (
            ([
                -u_reach_id(p),
                -self.trans_id(p, a, q),
                +ua_reach_id(q),
            ], None)
            for p in range(self.N)
            for q in range(self.N)
        )

        self._add_clauses(
            transition_state_clauses,
        )

        return ua_reach_id

    def generate_clauses_word(self, word):
        """
            Generate missing prefixes clauses.

            :return: word reach state ids
            :rtype: function(q:int):int
        """
        node = self.classification
        for letter in word:
            if letter not in node.keys():
                node[letter] = Classification(
                    label = self.generate_clauses_word_trans(node.label, letter),
                )
            node = node[letter]
        return node.label

    def generate_clauses_trace(self, word, positive=True):
        """
            :return: handle variable for this constraint
            :rtype: int
        """
        positive = bool(positive)

        word_reach_id = self.generate_clauses_word(word)
        handle_id = self._new_var()

        consistency_clauses = (
            ([
                -handle_id,
                -word_reach_id(q),
                (+1 if positive else -1) * self.term_id(q),
            ], None)
            for q in range(self.N)
        )

        self._add_clauses(
            consistency_clauses,
        )

        return handle_id

    def generate_clauses_reached_costates(self, dfa, coreach_id=None):
        """
            coreach_id(q,q2) will be True if ever reached.
            Creates N * len(dfa.states) new vars.

            :return: reach costate ids
            :rtype: function(int, int)
        """
        if coreach_id is None:
            coreach_ids = self._new_vars(self.N * len(dfa.states))
            coreach_id = lambda q, q2: coreach_ids[(q2) * self.N + q]

        initial_costate_clauses = (
            ([
                +coreach_id(0, dfa.states.index(init2)),
            ], None)
            for init2 in dfa._initial_states() # should be only one
        )

        transition_costate_clauses = (
            ([
                -coreach_id(p, p2),
                -self.trans_id(p, a, q),
                +coreach_id(q, dfa.states.index(next_state2)),
            ], None)
            for p in range(self.N)
            for p2,state2 in enumerate(dfa.states)
            for a,letter in enumerate(self.alphabet)
            for q in range(self.N)
            for next_state2 in dfa._next_states([state2], letter)
        )

        self._add_clauses(
            initial_costate_clauses,
            transition_costate_clauses,
        )

        return coreach_id


    def generate_clauses_hint(self, dfa, positive=True):
        """
            :return: handle variable for this constraint
            :rtype: int
        """
        positive = bool(positive)

        coreach_id = self.generate_clauses_reached_costates(dfa)
        handle_id = self._new_var()

        if positive: states_of_interest = dfa._terminal_states()
        else:        states_of_interest = dfa._states() - dfa._terminal_states()

        consistency_clauses = (
            ([
                -handle_id,
                -coreach_id(q, q2),
                (+1 if positive else -1) * self.term_id(q)
            ], None)
            for q in range(self.N)
            for q2,state2 in enumerate(dfa.states)
            if state2 in states_of_interest
        )

        self._add_clauses(
            consistency_clauses,
        )

        return handle_id



    def add_positive_traces(self, traces, weight=None):
        """
            :param traces: these words should be accepted by the automaton.
            :type traces: list(list(<Letter>))

            :param weight: weight of the soft constraint (hard constraint if set to None).
            :type weight: integer or None
        """
        handle_ids = [self.generate_clauses_trace(trace, positive=True) for trace in traces]
        handle_clauses = (
            ([
                +handle_id,
            ], weight)
            for handle_id in handle_ids
        )
        self._add_clauses(handle_clauses)
        return handle_ids

    def add_negative_traces(self, traces, weight=None):
        """
            :param traces: these words should be rejected by the automaton.
            :type traces: list(list(<Letter>))

            :param weight: weight of the soft constraint (hard constraint if set to None).
            :type weight: integer or None
        """
        handle_ids = [self.generate_clauses_trace(trace, positive=False) for trace in traces]
        self._add_clauses(
            ([
                +handle_id,
            ], weight)
            for handle_id in handle_ids
        )
        return handle_ids

    def add_positive_hints(self, dfas, weight=None):
        """
            Add positive hints, also called Type B or sub hints.
            :param dfas: words accepted by thoses languages should be accepted by the automaton (sub language).
            :type dfas: list(DFA)

            :param weight: weight of the soft constraint (hard constraint if set to None).
            :type weight: integer or None
        """
        handle_ids = [self.generate_clauses_hint(dfa, positive=True) for dfa in dfas]
        self._add_clauses(
            ([
                +handle_id,
            ], weight)
            for handle_id in handle_ids
        )
        return handle_ids

    def add_negative_hints(self, dfas, weight=None):
        """
            Add negative hints, also called Type A or sup hints.
            :param dfas: words rejected by thoses languages should be rejected by the automaton (sup language).
            :type dfas: list(DFA)

            :param weight: weight of the soft constraint (hard constraint if set to None).
            :type weight: integer or None
        """
        handle_ids = [self.generate_clauses_hint(dfa, positive=False) for dfa in dfas]
        self._add_clauses(
            ([
                +handle_id,
            ], weight)
            for handle_id in handle_ids
        )
        return handle_ids


    def add_hints(self, *args, **kwargs): return self.add_negative_hints(*args, **kwargs)



    def build_cnf(self):
        """
            Generate all previously added constraints.
            This method is called automatically by ``solve()`` but can still
            be called by the user (e.g. to time the constraints generation).
        """
        if self.wcnf is None:
            self.wcnf = pysat.formula.WCNF()
            self.wcnf.atms = [] # FIXME in the orig code

        for clauses in self.all_clauses:
            for clause, weight in clauses:
                self.wcnf.append(clause, weight=weight)
        self.all_clauses.clear()

        return self.wcnf

    def _solve_unweighted(self, solver=pysat.solvers.Glucose3):
        """
            Basic SAT solver. No not care about hard/soft distinction.
            .. warning:: ``build_cnf`` should have been called before.
        """
        self.solver = solver(bootstrap_with=self.wcnf.unweighted().clauses)
        if not self.solver.solve(): return False
        self.model = self.solver.get_model()
        return True

    def _solve_FM(self):
        """
            MAXSAT solver: https://pysathq.github.io/docs/pysat.pdf#subsection.1.2.1
            .. warning:: ``build_cnf`` should have been called before.
        """
        from pysat.examples.fm import FM
        self.solver = FM(self.wcnf, verbose=0)
        if not self.solver.compute(): return False
        # print(self.solver.cost)
        self.model = list(self.solver.model)
        return True

    # def solve(self, solver=pysat.solvers.Glucose3): # NOT WORKING
    #     # https://pysathq.github.io/docs/pysat.pdf#subsection.1.2.4
    #     from pysat.examples.lbx importLBX
    #     lbx = LBX(wcnf, use_cld=True, solver_name='g3')
    #     for mcs in lbx.enumerate():
    #         lbx.block(mcs)
    #         print(mcs)

    def _solve_RC2(self):
        """
            MAXSAT solver: https://pysathq.github.io/docs/pysat.pdf#subsection.1.2.9
            .. warning:: ``build_cnf`` should have been called before.
        """
        from pysat.examples.rc2 import RC2
        with RC2(self.wcnf) as rc2:
            for m in rc2.enumerate():
                # print('model{0}has cost{1}'.format(m, rc2.cost))
                self.model = m
                return True
        return False

    def solve(self, method="rc2"):
        """
            Solve the SAT problem.

            :param method: "rc2"|"fm"|"gc3"
        """
        self.build_cnf()
        return {
            "gc3": functools.partial(self._solve_unweighted, solver=pysat.solvers.Glucose3),
            "fm":  self._solve_FM,
            "rc2": self._solve_RC2,
        }[method]()

    def get_automaton(self):
        """
            Extract results as a DFA.
            .. warning:: ``solve`` should have been called successfully before.
        """
        transitions = {}
        accepting_states = []
        for p in range(self.N):
            for a,letter in enumerate(self.alphabet):
                for q in range(self.N):
                    if self.trans_id(p,a,q) in self.model:
                        if (p,letter) in transitions.keys():
                            print("WARNING: automaton not deterministic (too many transitions)", file=sys.stderr)
                        transitions[(p,letter)] = q
                    # elif -self.trans_id(p,a,q) not in self.model:
                    #     print("WARNING: transition undetermined", file=sys.stderr)
                if (p,letter) not in transitions.keys():
                    print("WARNING: automaton not deterministic (missing transition)", file=sys.stderr)
            # transitions.append(trans)
            if self.term_id(p) in self.model:
                accepting_states.append(p)
        return DFA(
            alphabet = self.alphabet,
            states = list(range(self.N)),
            transitions = transitions,
            init_states = [0],
            accepting_states = accepting_states,
        )

    def get_unclassified_samples(self):
        """
            return to lists of words: negative an positive.
            .. warning:: ``solve`` should have been called successfully before.
        """
        unclassified_positive_sample = []
        unclassified_negative_sample = []
        for i,word in enumerate(self.positive_sample):
            posid = +(1+i)
            if -self.POSIT_VARID(posid) in self.model:
                unclassified_positive_sample.append(word)
        clauses = self.wcnf.unweighted().clauses
        for i,word in enumerate(self.negative_sample):
            negid = -(1+i)
            if +self.NEGAT_VARID(negid) in self.model:
                unclassified_negative_sample.append(word)
        return (unclassified_positive_sample, unclassified_negative_sample)
