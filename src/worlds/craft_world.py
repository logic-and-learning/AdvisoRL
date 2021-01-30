if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import *
import random, math, os
import numpy as np
import random

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class CraftWorldParams:
    def __init__(self, file_map, use_tabular_representation, consider_night, movement_noise = 0):
        self.file_map     = file_map
        self.use_tabular_representation = use_tabular_representation
        self.movement_noise = movement_noise
        self.consider_night = consider_night

class CraftWorld:

    def __init__(self, params):
        self.params = params
        self._load_map(params.file_map)
        self.movement_noise = params.movement_noise
        self.env_game_over = False
        # Adding day and night if need it
        self.consider_night = params.consider_night
        self.hour = 12
        if self.consider_night:
            self.sunrise = 5
            self.sunset  = 21

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent
        self.hour = (self.hour + 1)%24

        # MDP
        p = 0.9
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
        ni,nj = self._get_next_position(action_, self.movement_noise)
        
        # Interacting with the objects that is in the next position (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni,nj)

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    def _get_next_position(self, action, movement_noise):
        """
        Returns the position where the agent would be if we execute action
        """
        agent = self.agent
        ni,nj = agent.i, agent.j

        # without jumping
        direction = action
        cardinals = set([Actions.up, Actions.down, Actions.left, Actions.right])
        if direction in cardinals and random.random() < movement_noise:
            direction = random.choice(list(cardinals - set([direction])))
        

        # OBS: Invalid actions behave as NO-OP
        if direction == Actions.up   : ni-=1
        if direction == Actions.down : ni+=1
        if direction == Actions.left : nj-=1
        if direction == Actions.right: nj+=1

        
        return ni,nj

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


    def _is_night(self):
        return not(self.sunrise <= self.hour <= self.sunset)

    def _steps_before_dark(self):
        if self.sunrise - 1 <= self.hour <= self.sunset:
            return 1 + self.sunset - self.hour
        return 0 # it is night

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        # adding the is_night proposition
        if self.consider_night and self._is_night():
            ret += "n"
        return ret

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        # if self.params.use_tabular_representation:
        #     return self._get_features_one_hot_representation()
        # return self._get_features_manhattan_distance()
        return self._get_features_one_hot_representation()


    def _get_features_manhattan_distance(self):
        # map from object classes to numbers
        class_ids = self.class_ids #{"a":0,"b":1}
        N,M = self.map_height, self.map_width
        ret = []
        for i in range(N):
            for j in range(M):
                obj = self.map_array[i][j]
                if str(obj) in class_ids:
                    ret.append(self._manhattan_distance(obj))
        
        # Adding the number of steps before night (if need it)
        if self.consider_night:
            ret.append(self._steps_before_dark())

        return np.array(ret, dtype=np.float64)


    """
    Returns the Manhattan distance between 'obj' and the agent
    """
    def _manhattan_distance(self, obj):
        return abs(obj.i - self.agent.i) + abs(obj.j - self.agent.j)

    """
    Returns a one-hot representation of the state (useful for the tabular case)
    """
    def _get_features_one_hot_representation(self):
        if self.consider_night:
            N,M,T = self.map_height, self.map_width, self.sunset - self.sunrise + 3
            ret = np.zeros((N,M,T), dtype=np.float64)
            ret[self.agent.i,self.agent.j, self._steps_before_dark()] = 1
        else:
            N,M = self.map_height, self.map_width
            ret = np.zeros((N,M), dtype=np.float64)
            ret[self.agent.i,self.agent.j] = 1
        return ret.ravel() # from 3D to 1D (use a.flatten() is you want to copy the array)

    # The following methods create a string representation of the current state ---------
    """
    Prints the current map
    """
    def show_map(self):
        print(self.__str__())
        if self.consider_night:
            print("Steps before night:", self._steps_before_dark(), "Current time:", self.hour)

    def __str__(self):
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i,j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if(i > 0):
                r += "\n"
            r += s
        return r

    # The following methods create the map ----------------------------------------------
    def _load_map(self,file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map (no monsters and no agent)
                - e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
            - self.agent: is the agent!
            - self.map_height: number of rows in every room 
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        # loading the map
        self.map_array = []
        self.class_ids = {} # I use the lower case letters to define the features
        f = open(file_map)
        i,j = 0,0
        for l in f:
            # I don't consider empty lines!
            if(len(l.rstrip()) == 0): continue
            
            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i,j,label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A":  entity = Empty(i,j)
                if e == "X":    entity = Obstacle(i,j)
                if e == "A":    self.agent = Agent(i,j,actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])



def play(params, task, max_time):
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    # play the game!
    game = CraftWorld(params)        
    rm = RewardMachine(task) 
    s1 = game.get_state()
    u1 = rm.get_initial_state()
    for t in range(max_time):
        # Showing game
        game.show_map()
        #print(game.get_features())
        #print(game.get_features().shape)
        #print(game._get_features_manhattan_distance())
        acts = game.get_actions()
        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action and str_to_action[a] in acts:
            game.execute_action(str_to_action[a])

            s2 = game.get_state()
            events = game.get_true_propositions()
            u2 = rm.get_next_state(u1, events)
            reward = rm.get_reward(u1,u2,s1,a,s2)

            if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                break 
            
            s1, u1 = s2, u2
        else:
            print("Forbidden action")
    game.show_map()
    return reward


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    map = "../../experiments/craft/maps/map_0.map"
    tasks = ["../../experiments/craft/reward_machines/t%d.txt"%i for i in [1,2,3,4,5,6,7,8,9,10]]
    max_time = 100
    use_tabular_representation=True
    consider_night=False

    for task in tasks:
        while True:
            params = CraftWorldParams(map, use_tabular_representation, consider_night)
            if play(params, task, max_time) > 0:
                break
