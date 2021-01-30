#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import itertools, functools, operator

class Classification(dict):
    """
        Tree structure allowing to classify words with labels (``None`` by default).
    """


    def __init__(self, label=None, *arg, **kwd):
        super(Classification, self).__init__(*arg, **kwd)
        self.label = label


    # def append(self, word,
    #     label = None,
    #     on_create = (lambda node, letter: None),
    # ):
    #     """
    #         :type word:  sequence(<Letter>)
    #         :type label: <Label>
    #         :rtype:      <Label>
    #     """
    #     w = iter(word)
    #     try:
    #         letter = next(w)
    #         if letter not in self.keys():
    #             self[letter] = self.__class__()
    #             on_create(self, letter)
    #         return self[letter].append(w, label)
    #     except StopIteration: # end of recursion
    #         old_label, self.label = self.label, label
    #         return old_label
    #
    # def append2(self, word,
    #     label = None,
    #     on_create = (lambda node, letter: None),
    # ):
    #     node = self
    #     for letter in word:
    #         if letter not in node.keys():
    #             node[letter] = self.__class__()
    #             on_create(node, letter)
    #         node = node[letter]
    #     old_label, node.label = node.label, label
    #     return old_label

    def get(self, word):
        """
            Get a word's label. Return ``None`` if the word is not classified.
            :type word: sequence(<Letter>)
            :rtype:     <Label>
        """
        w = iter(word)
        try:
            letter = next(w)
            if not letter in self.keys(): return None
            return self[letter].get(w)
        except StopIteration: # end of recursion
            return self.label


    def map_reduce(self, *args,
                   map        = (lambda label, *args: None),
                   reduce     = (lambda value, suffix_reduced: None),
                   args_rec   = (lambda letter, *args: args),
    ):
        """
            Iter over the tree structure.

            :param map:      function used on each node of the tree.
            :param reduce:   function used to merge the result of ``map`` and of each suffixes.
            :param args_rec: function used to change the arguments when going through branches.
            :return:         the result of the map-reduce operation.

            :type map:      function(<Label>, *list(<Args>)): <Value>
            :type reduce:   function(<Value>, <Value>): <Value>
            :type args_rec: function(<Letter>, *list(<Args>)): list(<Args>)
            :rtype:         <Value>
        """
        return functools.reduce(
            reduce,
            (
                suffix.map_reduce(
                    *args_rec(letter, *args),
                    args_rec=args_rec,
                    map=map,
                    reduce=reduce,
                )
                for letter, suffix in self.items()
            ),
            map(self.label, *args),
        )

    # without recursion (TODO)
    def map_reduce2(self, *args,
                   map        = (lambda label, *args: None),
                   reduce     = (lambda value, suffix_reduced: None),
                   args_rec   = (lambda letter, *args: args),
    ):
        """
            Iter over the tree structure.

            :param map:      function used on each node of the tree.
            :param reduce:   function used to merge the result of ``map`` and of each suffixes.
            :param args_rec: function used to change the arguments when going through branches.
            :return:         the result of the map-reduce operation.

            :type map:      function(<Label>, *list(<Args>)): <Value>
            :type reduce:   function(<Value>, <Value>): <Value>
            :type args_rec: function(<Letter>, *list(<Args>)): list(<Args>)
            :rtype:         <Value>
        """
        return functools.reduce(
            reduce,
            (
                suffix.map_reduce(
                    *args_rec(letter, *args),
                    args_rec=args_rec,
                    map=map,
                    reduce=reduce,
                )
                for letter, suffix in self.items()
            ),
            map(self.label, *args),
        )

    def print_tree(self, file=sys.stdout, print_unclassified=False):
        self.map_reduce("",
            map = (lambda label, prefix:
                file.write("{}: {}\n".format(prefix, label))
                if (print_unclassified or label is not None) else None
            ),
            args_rec = (lambda letter, prefix:
                ["{}{}".format(prefix, letter)]
            ),
        )


    def count_words(self, count_unclassified=False, *, filter=None, label=None):
        if filter is None: filter = lambda l: True
        if label is not None: filter = lambda l: l==label
        return self.map_reduce(
            map = (lambda l: bool((l is not None or count_unclassified) and filter(l))),
            reduce = operator.add,
        )


    def max_len(self):
        return self.map_reduce(
            map = (lambda label: 0),
            reduce = (lambda v,s: max(v,s+1)),
        )


    def to_simple_dict(self):
        def reduce(tree, branch):
            tree[1][branch[0]] = branch[1]
            return tree
        return self.map_reduce(None,
            map = (lambda label, l: (l,{})),
            reduce = reduce,
            args_rec = (lambda letter, l: [letter]),
        )[1]
