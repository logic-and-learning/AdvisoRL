"""
    inspired from https://gym.openai.com/envs/Taxi-v3/
"""

if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import *
from automata_learning.Traces import Traces
import random, math, os
import numpy as np
import random

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class TaxiWorldParams:
    def __init__(self, file_map):
        self.file_map = file_map

class TaxiWorld:

    def __init__(self, params):
        self.params = params
        self._load_map(params.file_map)
        self.env_game_over = False

    def execute_action(self, a):
        """
            We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent

        # MDP
        # p = 0.9
        p = 1.0 # desactivate slip
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        if (check<=slip_p[0]):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0:
                a_ = 3
            elif a == 2:
                a_ = 1
            elif a == 3:
                a_ = 2
            elif a == 1:
                a_ = 0

        else:
            if a == 0:
                a_ = 1
            elif a == 2:
                a_ = 3
            elif a == 3:
                a_ = 0
            elif a == 1:
                a_ = 2

        action_ = Actions(a_)
        self.a_ = a_

        # Getting new position after executing action
        ni,nj = agent.i, agent.j
        action_ = Actions(a_)
        self.a_ = a_
        if (ni,nj,action_) not in self.forbidden_transitions:
            if action_ == Actions.up   : ni-=1
            if action_ == Actions.down : ni+=1
            if action_ == Actions.left : nj-=1
            if action_ == Actions.right: nj+=1
        current_loc = self.objects.get((ni,nj), "")
        if action_ == Actions.none: # pickup
            if self.passenger == current_loc:
                self.passenger = None # in taxi
            else:
                pass # reward -= 10
                # self.env_game_over = True
        if action_ == Actions.drop: # dropoff
            if self.passenger == None and current_loc:
                # if current_loc == self.destination: self.passenger = current_loc
                self.passenger = current_loc
                # if current_loc == self.destination:
                #     pass # reward += 20
            else:
                pass # reward -= 10
                # self.env_game_over = True

        agent.change_position(ni,nj)

    def get_state(self):
        return None # we are only using "simple reward machines" for the taxi domain



    def get_actions(self):
        """
            Returns the list with the actions that the agent can perform
        """
        return self.agent.get_actions()

    def get_last_action(self):
        """
            Returns agent's last action
        """
        return self.a_

    def get_true_propositions(self):
        """
            Returns the string with the propositions that are True in this state
        """
        current_loc = self.objects.get((self.agent.i,self.agent.j), "")
        if current_loc: # taxi at location
            loc_i = "abcd".index(current_loc.lower())
            if self.passenger == current_loc: # passenger at location
                return Traces.letters[loc_i+4]
            else: # passenger in taxi or elsewere
                return Traces.letters[loc_i]
        else: # taxi in transit
            return ""

        # ret = self.objects.get((self.agent.i,self.agent.j), "").lower()
        # ret += "efgh"["abcd".index(self.destination.lower())]
        # if self.passenger is not None: # at location
        #     ret += "ijkl"["abcd".index(self.passenger.lower())]
        # else: # in taxi
        #     ret += "m"
        # return ret

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        N,M = self.map_height, self.map_width
        ret = np.zeros((N,M), dtype=np.float64)
        ret[self.agent.i,self.agent.j] = 1
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)
        # if self.params.use_tabular_representation:
        #     return self._get_features_one_hot_representation()
        # return self._get_features_manhattan_distance()
        # return self._get_features_one_hot_representation()


    # def _get_features_manhattan_distance(self):
    #     # map from object classes to numbers
    #     class_ids = self.class_ids #{"a":0,"b":1}
    #     N,M = self.map_height, self.map_width
    #     ret = []
    #     for i in range(N):
    #         for j in range(M):
    #             obj = self.map_array[i][j]
    #             if str(obj) in class_ids:
    #                 ret.append(self._manhattan_distance(obj))
    #
    #     # Adding the number of steps before night (if need it)
    #     if self.consider_night:
    #         ret.append(self._steps_before_dark())
    #
    #     return np.array(ret, dtype=np.float64)


    # def _manhattan_distance(self, obj):
    #     """
    #         Returns the Manhattan distance between 'obj' and the agent
    #     """
    #     return abs(obj.i - self.agent.i) + abs(obj.j - self.agent.j)
    #
    # def _get_features_one_hot_representation(self):
    #     """
    #         Returns a one-hot representation of the state (useful for the tabular case)
    #     """
    #     if self.consider_night:
    #         N,M,T = self.map_height, self.map_width, self.sunset - self.sunrise + 3
    #         ret = np.zeros((N,M,T), dtype=np.float64)
    #         ret[self.agent.i,self.agent.j, self._steps_before_dark()] = 1
    #     else:
    #         N,M = self.map_height, self.map_width
    #         ret = np.zeros((N,M), dtype=np.float64)
    #         ret[self.agent.i,self.agent.j] = 1
    #     return ret.ravel() # from 3D to 1D (use a.flatten() is you want to copy the array)

    # The following methods create a string representation of the current state ---------

    def show_map(self):
        """
            Prints the current map
        """
        print(self.__str__())

    def __str__(self):
        r = "+" + "-"*(self.map_width*2-1) + "+\n"
        for i in range(self.map_height):
            s = "|"
            for j in range(self.map_width):
                if self.agent.idem_position(i,j):
                    # s += str(self.agent)
                    s += "T"
                else:
                    s += str(self.objects.get((i,j), " "))
                if (i,j,Actions.right) in self.forbidden_transitions:
                    s += "|"
                else:
                    s += ":"
            r += s + "\n"
        r += "+" + "-"*(self.map_width*2-1) + "+"
        return r

    # The following methods create the map ----------------------------------------------
    def _load_map(self,file_map):
        """
            This method adds the following attributes to the game:
                - self.objects: dict of features
                - self.forbidden_transitions: set of forbidden transitions (i,j,a)
                - self.agent: is the agent!
                - self.map_height: number of rows in every room
                - self.map_width: number of columns in every room
            The inputs:
                - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        actions = [
            Actions.down.value,  # move south
            Actions.up.value,    # move north
            Actions.left.value,  # move east
            Actions.right.value, # move west
            Actions.none.value,  # pickup passenger
            Actions.drop.value,  # dropoff passenger
        ]

        self.objects = {}
        self.forbidden_transitions = set()
        with open(file_map) as f:
            map = [line.rstrip()
                for line in f.readlines()
                if line.rstrip() # skip empty lines
                if not "-" in line # skip beginning and end
            ]
        # loading the map
        for i,line in enumerate(map):
            for j,c in enumerate(range(1,len(line),2)):
                e = line[c]
                if e not in " ":
                    self.objects[(i,j)] = e
                if line[c-1] == "|": self.forbidden_transitions.add((i,j,Actions.left))
                if line[c+1] == "|": self.forbidden_transitions.add((i,j,Actions.right))
                if i == 0:           self.forbidden_transitions.add((i,j,Actions.up))
                if i == len(map)-1:  self.forbidden_transitions.add((i,j,Actions.down))
        self.map_height, self.map_width = i+1, j+1 # last i and j used

        while True:
            i, j = random.randrange(self.map_height), random.randrange(self.map_width)
            if (i,j) not in self.objects.keys(): break # prevent the taxi spawning on a location
            # break
        self.agent = Agent(i,j,actions)

        self.passenger   = "A"
        # self.passenger   = random.choice([obj for obj in self.objects.values()])
        # self.destination = random.choice([obj for obj in self.objects.values() if obj!=self.passenger]) # defined in the task

#TODO
# def play(params, task, max_time):
#     from reward_machines.reward_machine import RewardMachine
#
#     # commands
#     str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
#     # play the game!
#     game = TaxiWorld(params)
#     rm = RewardMachine(task)
#     s1 = game.get_state()
#     u1 = rm.get_initial_state()
#     for t in range(max_time):
#         # Showing game
#         game.show_map()
#         #print(game.get_features())
#         #print(game.get_features().shape)
#         #print(game._get_features_manhattan_distance())
#         acts = game.get_actions()
#         # Getting action
#         print("\nAction? ", end="")
#         a = input()
#         print()
#         # Executing action
#         if a in str_to_action and str_to_action[a] in acts:
#             game.execute_action(str_to_action[a])
#
#             s2 = game.get_state()
#             events = game.get_true_propositions()
#             u2 = rm.get_next_state(u1, events)
#             reward = rm.get_reward(u1,u2,s1,a,s2)
#
#             if game.env_game_over or rm.is_terminal_state(u2): # Game Over
#                 break
#
#             s1, u1 = s2, u2
#         else:
#             print("Forbidden action")
#     game.show_map()
#     return reward


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    map = "../../experiments/taxi/maps/map_0.map"
    tasks = ["../../experiments/taxi/reward_machines/t%d.txt"%i for i in [1,2,3,4,5,6,7,8,9,10]]
    max_time = 100

    for task in tasks:
        while True:
            params = TaxiWorldParams(map)
            if play(params, task, max_time) > 0:
                break
