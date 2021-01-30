from tester.tester_craft import TesterCraftWorld
from tester.tester_office import TesterOfficeWorld
from tester.tester_traffic import TesterTrafficWorld
from reward_machines.reward_machine import RewardMachine
from tester.test_utils import read_json, get_precentiles_str, get_precentiles_in_seconds, reward2steps
import numpy as np
import time, os
import matplotlib.pyplot as plt
#import pdb

class TesterPolicyBank:
    def __init__(self, learning_params, testing_params, experiment, result_file=None):
        if result_file is None: # in this case, we are running a new experiment
            self.learning_params = learning_params
            self.testing_params = testing_params
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()

            # setting the right world environment
            self.game_type = eval(lines[0])
            if self.game_type == "officeworld":
                self.world = TesterOfficeWorld(experiment, learning_params.gamma)
            if self.game_type == "craftworld":
                self.world = TesterCraftWorld(experiment, learning_params.tabular_case, learning_params.gamma)
            if self.game_type == "trafficworld":
                self.world = TesterTrafficWorld(experiment, learning_params.tabular_case, learning_params.gamma)

            # Creating the reward machines for each task
            self.reward_machines = []
            self.file_to_reward_machine = {}
            rm_files = self.world.get_reward_machine_files()
            for i in range(len(rm_files)):
                rm_file = rm_files[i]
                self.file_to_reward_machine[rm_file] = i
                self.reward_machines.append(RewardMachine(rm_file))
            self.hypothesis_machine_file = "../src/automata_learning/hypothesis_machine.txt" #remove these lines? negligible when using separate testers
            self.hypothesis_machine = RewardMachine( self.hypothesis_machine_file ) #


            # I store the results here
            self.results = {}
            self.steps = []
            aux_tasks = self.get_task_specifications()
            for i in range(len(aux_tasks)):
                t_str = str(aux_tasks[i])
                self.results[t_str] = {}

        else:
            # In this case, we load the results that were precomputed in a previous run
            data = read_json(result_file)
            self.game_type = data['game_type']
            if self.game_type == "craftworld":
                self.world = TesterCraftWorld(None, None, None, data['world'])
            if self.game_type == "trafficworld":
                self.world = TesterTrafficWorld(None, None, data['world'])
            if self.game_type == "officeworld":
                self.world = TesterOfficeWorld(None, None, data['world'])

            self.results = data['results']
            self.steps   = data['steps']            
            # obs: json transform the interger keys from 'results' into strings
            # so I'm changing the 'steps' to strings
            for i in range(len(self.steps)):
                self.steps[i] = str(self.steps[i])

    def update_hypothesis_machine(self):
        self.hypothesis_machine = RewardMachine(self.hypothesis_machine_file)

    def update_hypothesis_machine_file(self,hmfile):
        self.hypothesis_machine_file = hmfile

    def get_hypothesis_machine(self):
        return self.hypothesis_machine

    def get_world_name(self):
        return self.game_type.replace("world","")

    def get_task_params(self, task_specification):
        return self.world.get_task_params(task_specification)

    def get_reward_machine_id_from_file(self, rm_file):
        return self.file_to_reward_machine[rm_file]

    def get_reward_machine_id(self, task_specification):
        rm_file = self.world.get_task_rm_file(task_specification)
        return self.get_reward_machine_id_from_file(rm_file)

    def get_reward_machines(self):
        return self.reward_machines

    def get_world_dictionary(self):
        return self.world.get_dictionary()
    
    def get_task_specifications(self):
        # Returns the list with the task specifications (reward machine + env params)
        return self.world.get_task_specifications()

    def get_task_rms(self):
        # Returns only the reward machines that we are learning
        return self.world.get_reward_machine_files()

    def get_optimal(self, task):
        r = self.world.optimal[task]
        return r if r > 0 else 1.0

    def run_test(self, step, sess, test_function, rm_learned, rm_true, is_learned, q, *test_args):
        t_init = time.time()
        # 'test_function' parameters should be (sess, task_params, learning_params, testing_params, *test_args)
        # and returns the reward
       # reward_machines = self.get_reward_machines()
        reward_machines = [self.get_hypothesis_machine()]
        aux = []
        for task_specification in self.get_task_specifications():
            task_str = str(task_specification)
            task_params = self.get_task_params(task_specification)
            task_rm_id  = self.get_reward_machine_id(task_specification)
            reward = test_function(sess, reward_machines, task_params, rm_learned, rm_true, is_learned, q, self.learning_params, self.testing_params, 14, *test_args)
            if step not in self.results[task_str]:
                self.results[task_str][step] = []
            if len(self.steps) == 0 or self.steps[-1] < step:
                self.steps.append(step)
            if reward is None:
                # the test returns 'none' when, for some reason, this network hasn't change
                # so we have to copy the results from the previous iteration
                id_step = [i for i in range(len(self.steps)) if self.steps[i] == step][0] - 1
                reward = 0 if id_step < 0 else self.results[task_str][self.steps[id_step]][-1]
                #print("Skiped reward is", reward)
            self.results[task_str][step].append(reward) 
            aux.append(reward)
        if self.game_type=="officeworld" or self.game_type=="craftworld" or self.game_type=="trafficworld":
            print("Testing: %0.1f"%(time.time() - t_init), "seconds\tTotal: %d"%sum([(r if r > 0 else self.testing_params.num_steps) for r in reward2steps(aux)]))
            print("\t".join(["%d"%(r) for r in reward2steps(aux)]))
        return reward

    def show_results(self):
        average_reward = {}
        
        tasks = self.get_task_specifications()

        # Showing perfomance per task
        for t in tasks:
            t_str = str(t)
            print("\n" + t_str + " --------------------")
            print("steps\tP25\t\tP50\t\tP75")            
            for s in self.steps:
                normalized_rewards = [r/self.get_optimal(t) for r in self.results[t_str][s]]
                a = np.array(normalized_rewards)
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]
                p25, p50, p75 = get_precentiles_str(a)
                #p25, p50, p75 = get_precentiles_in_seconds(a)
                print(str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)


        # arrays for plot
        a25 = np.empty([0,len(self.steps)])
        a50 = np.empty([0,len(self.steps)])
        a75 = np.empty([0,len(self.steps)])

        # Showing average perfomance across all the task
        print("\nAverage Reward --------------------")
        print("steps\tP25\t\tP50\t\tP75")
        num_tasks = float(len(tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            p25, p50, p75 = get_precentiles_str(normalized_rewards)
            a25 = np.append(a25,float(p25))
            a50 = np.append(a50,float(p50))
            a75 = np.append(a75,float(p75))
            p25, p50, p75 = get_precentiles_in_seconds(normalized_rewards)
            print(str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)
        self.plot_performance(self.steps,a25,a50,a75)

    def plot_performance(self,steps,p25,p50,p75):

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)
        ax.plot(steps, p25, alpha=0)
        ax.plot(steps, p50, color='black')
        ax.plot(steps, p75, alpha=0)
        ax.grid()
        plt.fill_between(steps, p50, p25, color='grey', alpha=0.25)
        plt.fill_between(steps, p50, p75, color='grey', alpha=0.25)
        ax.set_xlabel('number of steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.locator_params(axis='x', nbins=5)
        plt.gcf().subplots_adjust(bottom=0.15)
        ax.tick_params(axis='both', which='major', labelsize=22)
        plt.savefig('figure_1.png', dpi=600)
        plt.show()

    def plot_this(self,a1,a2):
        fig, ax = plt.subplots()
        ax.plot(a1,a2, color='blue')
        ax.grid()
        ax.set(xlabel='number of steps', ylabel='reward')
        plt.show()

    def get_best_performance_per_task(self):
        # returns the best performance per task (this is relevant for reward normalization)
        ret = {}
        for t in self.get_task_specifications():
            t_str = str(t)
            ret[t_str] = max([max(self.results[t_str][s]) for s in self.steps]) 
        return ret

    def get_result_summary(self):
        """
        Returns normalized average performance across all the tasks
        """
        average_reward = {}
        task_reward = {}
        task_reward_count = {}
        tasks = self.get_task_specifications()

        # Computing average reward per task
        for task_specification in tasks:
            t_str = str(task_specification)
            task_rm = self.world.get_task_rm_file(task_specification)
            if task_rm not in task_reward:
                task_reward[task_rm] = {}
                task_reward_count[task_rm] = 0
            task_reward_count[task_rm] += 1
            for s in self.steps:
                normalized_rewards = [r/self.get_optimal(t_str) for r in self.results[t_str][s]]
                a = np.array(normalized_rewards)
                # adding to the average reward
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]
                # adding to the average reward per tas
                if s not in task_reward[task_rm]: task_reward[task_rm][s] = a
                else: task_reward[task_rm][s] = a + task_reward[task_rm][s]

        # Computing average reward across all tasks
        ret = []
        ret_task = {}
        for task_rm in task_reward:
            ret_task[task_rm] = []
        num_tasks = float(len(tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            ret.append([s, normalized_rewards])
            for task_rm in task_reward:
                normalized_task_rewards = task_reward[task_rm][s] / float(task_reward_count[task_rm])
                ret_task[task_rm].append([s, normalized_task_rewards])
        ret_task["all"] = ret
                
        return ret_task

