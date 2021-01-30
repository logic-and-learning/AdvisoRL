import pdb
import random
class DFA:


    def __init__(self, RPNI_output_file_name):


        self.states = []
        self.alphabet = []
        self.transitions = {}
        self.accepting_states = []
        self.init_states = []

        with open(RPNI_output_file_name) as RPNI_output_file:
            mode = "general"
            for line in RPNI_output_file:
                if "alphabet size" in line:
                    size = int(line.split("=")[1].strip().strip(';'))
                    self.alphabet = list(range(size))
                if "number of states" in line:
                    num_states = int(line.split("=")[1].strip().strip(';'))
                    self.states = list(range(num_states))
                if "initial states" in line:
                    mode = "init"
                    continue
                if "final states" in line:
                    mode = "final"
                    continue
                if "transitions" in line:
                    mode = "transitions"
                    continue

                if mode == "init":
                    line = line.strip().strip(';')
                    listOfStates = line.split(',')
                    self.init_states = [int(s) for s in listOfStates]
                    if len(self.init_states) > 1:
                        raise ValueError("the automaton has more than 1 initial state")

                if mode == "final":
                    line = line.strip().strip(';')
                    listOfStates = line.split(',')
                    self.accepting_states = list()
                    for s in listOfStates:
                        if s!= '':
                            self.accepting_states.append(int(s))
                    if self.accepting_states=='':
                        self.accepting_states.append(int(random.choice(range(0,51))))

                if mode == "transitions":
                    line = line.strip().strip(';')
                    transition_description = line.split(',')
                    self.transitions[(int(transition_description[0]), int(transition_description[1]))] = int(transition_description[2])


    def export_as_reward_automaton(self, output_file_name):
        with open(output_file_name, "w") as output_file:
            output_file.write(str(self.init_states[0])+" # initial state")
            for trans in self.transitions:
                output_file.write("\n")
                if self.transitions[trans] in self.accepting_states:
                    reward = 1
                else:
                    reward = 0

                startingState = trans[0]
                goalState = self.transitions[trans]
                symbol = trans[1]

                reward_description = "ConstantRewardFunction("+str(reward)+")"

                output_file.write( "("+str(startingState)+","+str(goalState)+",'"+str(symbol) + "', "+reward_description + ")")



