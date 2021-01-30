if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import ActionsNew
import random, math, os
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class TrafficWorldParams:
    def __init__(self, file_map, use_tabular_representation, consider_night, movement_noise = 0):
        self.file_map     = file_map
        self.use_tabular_representation = use_tabular_representation


class TrafficWorld:

    def __init__(self, params):
        self._load_map()
        self.env_game_over = False

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        x,y = self.agent
        # executing action
        self.agent = self.xy_MDP_slip(a,1) # progresses in x-y system

    def xy_MDP_slip(self,a,p):
        x,y = self.agent
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        # forward = 0
        # left  = 1
        # right = 2
        # stay  = 3


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

        action_ = ActionsNew(a_)

        x+=1
        y+=1

        if (x-1,y-1,action_) in self.forbidden_transitions:
            action_ = ActionsNew.forward

        if True:
            if action_ == ActionsNew.forward and (y==3 or y==8) and (x==2 or x==7):
                x+=3
            elif action_ == ActionsNew.forward and (y==3 or y==8) and (x!=2 and x!=7 and x!=11):
                x+=1
            elif action_ == ActionsNew.forward and (y==4 or y==9) and (x==5 or x==10):
                x-=3
            elif action_ == ActionsNew.forward and (y==4 or y==9) and (x!=5 and x!=10 and x!=1):
                x-=1
            elif action_ == ActionsNew.forward and (x==4 or x==9) and (y==2 or y==7):
                y+=3
            elif action_ == ActionsNew.forward and (x==4 or x==9) and (y!=2 and y!=7 and y!=11):
                y+=1
            elif action_ == ActionsNew.forward and (x==3 or x==8) and (y==5 or y==10):
                y-=3
            elif action_ == ActionsNew.forward and (x==3 or x==8) and (y!=5 and y!=10 and y!=1):
                y-=1
            elif action_ == ActionsNew.forward and ( (x == 1 and y == 4) or (x == 1 and y == 9) ):
                y-=1
            elif action_ == ActionsNew.forward and ( (x == 11 and y == 3) or (x == 11 and y == 8) ):
                y+=1
            elif action_ == ActionsNew.forward and ( (x == 3 and y == 1) or (x == 8 and y == 1) ):
                x+=1
            elif action_ == ActionsNew.forward and ( (x == 4 and y == 11) or (x == 9 and y == 11) ):
                x-=1
            elif action_ == ActionsNew.left and ( (x==2 and y==3) or (x==7 and y==3) or (x==2 and y==8) or (x==7 and y==8) ):
                x+=2
                y+=2
            elif action_ == ActionsNew.right and ( (x==2 and y==3) or (x==7 and y==3) or (x==2 and y==8) or (x==7 and y==8) ):
                x+=1
                y-=1
            elif action_ == ActionsNew.left and ( (x==4 and y==2) or (x==9 and y==2) or (x==4 and y==7) or (x==9 and y==7) ):
                x-=2
                y+=2
            elif action_ == ActionsNew.right and ( (x==4 and y==2) or (x==9 and y==2) or (x==4 and y==7) or (x==9 and y==7) ):
                x+=1
                y+=1
            elif action_ == ActionsNew.left and ( (x==5 and y==4) or (x==10 and y==4) or (x==5 and y==9) or (x==10 and y==9) ):
                x-=2
                y-=2
            elif action_ == ActionsNew.right and ( (x==5 and y==4) or (x==10 and y==4) or (x==5 and y==9) or (x==10 and y==9) ):
                x-=1
                y+=1
            elif action_ == ActionsNew.left and ( (x==3 and y==5) or (x==8 and y==5) or (x==3 and y==10) or (x==8 and y==10) ):
                x+=2
                y-=2
            elif action_ == ActionsNew.right and ( (x==3 and y==5) or (x==8 and y==5) or (x==3 and y==10) or (x==8 and y==10) ):
                x-=1
                y-=1


        x-=1
        y-=1

        self.a_ = a_
        return (x,y)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.a_

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_true_propositions_action(self,a):
        """
        Returns the string with the propositions that are True in this state
        """
        # forward = 0
        # left = 1
        # right = 2
        # stay = 3
        # a stop and in priority road
        # b stop and not in priority road
        # c not stop and in priority road
        # d not stop and not in priority road
        # e B

        ret = ""
        if self.agent in self.points:
            if self.points[self.agent]==1:
                if a==3:
                    ret += "a"
                elif a==0:
                    ret += "c"
                else:
                    ret += "d"
            if self.points[self.agent]==2:
                if a==3:
                    ret += "b"
                elif a==0:
                    ret += "d"
                else:
                    ret += "c"

        if self.agent == (10,2):
            ret += "e"
        # if self.agent in self.objects:
        #     ret += self.objects[self.agent]
        return ret

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        x,y = self.agent
        N,M = 11,11
        ret = np.zeros((N,M), dtype=np.float64)
        ret[x,y] = 1
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)

    # The following methods create the map ----------------------------------------------
    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.points = {}

        self.points[(1, 2)] = 1
        self.points[(1, 7)] = 1
        self.points[(6, 2)] = 1
        self.points[(6, 7)] = 1
        self.points[(4, 3)] = 1
        self.points[(4, 8)] = 1
        self.points[(9, 3)] = 1
        self.points[(9, 8)] = 1

        self.points[(2, 4)] = 2
        self.points[(2, 9)] = 2
        self.points[(7, 4)] = 2
        self.points[(7, 9)] = 2
        self.points[(3, 1)] = 2
        self.points[(3, 6)] = 2
        self.points[(8, 1)] = 2
        self.points[(8, 6)] = 2

        self.objects[(10,2)] = "b"

        # Adding walls
        self.forbidden_transitions = set()
        # general grid
        for x in [1,5,6,10,11]:
            for y in [3,8]:
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.left))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.right))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.stay))
        for x in [1,2,6,7,11]:
            for y in [4,9]:
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.left))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.right))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.stay))
        for y in [1,2,6,7,11]:
            for x in [3,8]:
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.left))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.right))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.stay))
        for y in [1,5,6,10,11]:
            for x in [4,9]:
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.left))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.right))
                self.forbidden_transitions.add((x-1,y-1, ActionsNew.stay))
        # Adding the agent
        self.agent = (0,7)
        self.actions = [ActionsNew.forward.value,ActionsNew.right.value,ActionsNew.stay.value,ActionsNew.left.value]

