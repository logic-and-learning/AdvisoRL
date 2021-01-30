if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import Actions
import random, math, os
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class OfficeWorldParams:
    def __init__(self):
        pass

class OfficeWorld:

    def __init__(self, params):
        self._load_map()
        self.env_game_over = False

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        x,y = self.agent
        # executing action
        self.agent = self.xy_MDP_slip(a,0.9) # progresses in x-y system

    def xy_MDP_slip(self,a,p):
        x,y = self.agent
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

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
        if (x,y,action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                y+=1
            if action_ == Actions.down:
                y-=1
            if action_ == Actions.left:
                x-=1
            if action_ == Actions.right:
                x+=1

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

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        x,y = self.agent
        N,M = 12,9
        ret = np.zeros((N,M), dtype=np.float64)
        ret[x,y] = 1
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)


    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    # The following methods create the map ----------------------------------------------
    def _load_map_old(self):
        # Creating the map
        self.objects = {}
        #env.agent = tuple([2, 2])
        #env.coffee = tuple([3, 5])
        #env.init_agent = tuple([2, 2])
        #env.locations = {(1, 1): 'a', (10, 1): 'b', (7, 3): 'c', (7, 4): 'e', (3, 5): 'f', (4, 4): 'g', (1, 8): 'd'}
        #env.mail = tuple([7, 4])
        self.objects[(1,1)] = "a"
        self.objects[(10,1)] = "b"
        #self.objects[(10,7)] = "c"
        self.objects[(1, 3)] = "c"
        #self.objects[(1,7)] = "d"
        self.objects[(7,4)] = "e"  # MAIL
        #self.objects[(8,2)] = "f"  # COFFEE
        self.objects[(3,5)] = "f"  # COFFEE
        self.objects[(4,4)] = "g"  # OFFICE

        # Adding walls
        self.forbidden_transitions = set()
        # for x in range(12):
        #     for y in [0]:
        #         self.forbidden_transitions.add((x,y,Actions.down))
        #     for y in [8]:
        #         self.forbidden_transitions.add((x,y,Actions.up))
        # for y in range(9):
        #     for x in [0]:
        #         self.forbidden_transitions.add((x,y,Actions.left))
        #     for x in [11]:
        #         self.forbidden_transitions.add((x,y,Actions.right))
        # general grid
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down))
                self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
            for x in [0,3,6,9]:
                self.forbidden_transitions.add((x,y,Actions.left))
                self.forbidden_transitions.add((x+2,y,Actions.right))
        # adding 'doors'
        for y in [1,7]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        # Adding the agent
        self.agent = (2,1)
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]
    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(1,1)] = "a"
        self.objects[(10,1)] = "b"
        #self.objects[(10,7)] = "c"
        self.objects[(1, 3)] = "c"
        self.objects[(1,7)] = "d"
        self.objects[(7,4)] = "e"  # MAIL
        self.objects[(3,5)] = "f"  # COFFEE
        self.objects[(4,4)] = "g"  # OFFICE

        # Adding the agent
        self.agent = (2,1)
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]

        # Adding walls
        self.forbidden_transitions = set()
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down))
                self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
            for x in [0,3,6,9]:
                self.forbidden_transitions.add((x,y,Actions.left))
                self.forbidden_transitions.add((x+2,y,Actions.right))
        # adding 'doors'
        for y in [1,7]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        
def play():
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    params = OfficeWorldParams()

    # play the game!
    tasks = ["../../experiments/office/reward_machines/t%d.txt"%i for i in [1,2,3,4]]
    reward_machines = []
    for t in tasks:
        reward_machines.append(RewardMachine(t))
    for i in range(len(tasks)):
        print("Running", tasks[i])

        game = OfficeWorld(params) # setting the environment
        rm = reward_machines[i]  # setting the reward machine
        s1 = game.get_state()
        u1 = rm.get_initial_state()
        while True:
            # Showing game
            game.show()
            print("Events:", game.get_true_propositions())
            #print(game.getLTLGoal())
            # Getting action
            print("u:", u1)
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                game.execute_action(str_to_action[a])

                # Getting new state and truth valuation
                s2 = game.get_state()
                events = game.get_true_propositions()
                u2 = rm.get_next_state(u1, events)
                r = rm.get_reward(u1,u2,s1,a,s2)
                
                # Getting rewards and next states for each reward machine
                rewards, next_states = [],[]
                for j in range(len(reward_machines)):
                    j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
                    rewards.append(j_rewards)
                    next_states.append(j_next_states)
                
                print("---------------------")
                print("Rewards:", rewards)
                print("Next States:", next_states)
                print("Reward:", r)
                print("---------------------")
                
                if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                    break 
                
                s1 = s2
                u1 = u2
            else:
                print("Forbidden action")
        game.show()
        print("Events:", game.get_true_propositions())
    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()
