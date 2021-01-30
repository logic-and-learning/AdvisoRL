import numpy as np
import random, time
import tensorflow as tf
from automata_learning_utils import al_utils
from worlds.game import *
from automata_learning_with_policybank.policy_bank_dqn import PolicyBankDQN
from common.schedules import LinearSchedule
from common.replay_buffer import create_experience_replay_buffer
from automata_learning_with_policybank.Traces import Traces
from tester_policybank.tester import TesterPolicyBank as Tester
import qrm
import shutil
import os
import subprocess
import csv

#import pdb

######## compare rm_learned & rm_true to find conflicting experiences
######## check is_rm_learned in run_aqrm_task and break all operations if rewards don't match

######## make new tester tester_current with past and new experience
######## pass tester_current.get_reward_machines as arguments to decompose_reward_machines
######## make new tester
######## make copies of pertinent reward machines to some file path
######## write new experiment file sifting through each of the reward machines

def run_aqrm_task(sess, epsilon, environment_rm_file, learned_rm_file, policy_bank, tester_true, tester_learned, curriculum, replay_buffer, beta_schedule, show_print, is_rm_learned, currentstep, previous_testing_reward, q):
    """
    This code runs one training episode. 
        - rm_file: It is the path towards the RM machine to solve on this episode
        - environment_rm: an environment reward machine, the "true" one, underlying the execution
    """
    # Initializing parameters and the game
    learning_params = tester_learned.learning_params
    testing_params = tester_learned.testing_params

    """
     here, tester holds all the machines. we would like to dynamically update the machines every so often.
     an option might be to read it every time a new machine is learnt
     """
    reward_machines = [tester_learned.get_hypothesis_machine()]

    task_params = tester_learned.get_task_params(learned_rm_file) # rm_files redundant here unless in water world (in which case it provides the map files based on the task)
    rm_true = tester_true.get_reward_machines()[0] # add one more input n to track tasks at hand, replace 0 with n
    rm_learned = tester_learned.get_hypothesis_machine()

    task = Game(task_params)
    actions = task.get_actions()
    ok = 1
    num_features = len(task.get_features())
    num_steps = learning_params.max_timesteps_per_task
    training_reward = 0
    is_conflicting=1 #by default add traces
    testing_reward = None #initialize

    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    u1 = rm_learned.get_initial_state()
    u1_true = rm_true.get_initial_state()
    has_been = [0,0]
    alpha = 0.8
    gamma = 0.99
    w = 0

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)
    all_events = []
    sy_s = [[]]
    a_s = []
    a=0
    for t in range(num_steps):
        currentstep += 1
        s = np.where(s1_features==1)[0][0]
        # sy = s%11+1
        # sx = (s-sy+1)/11+1
        # sy_s.append([sx,sy])
        # a_s.append(a)
        # Choosing an action to perform
        if random.random() < 0.15:
            a = random.choice(actions)
        else:
            #IG: current problem: there is no machine so a default behavior is to stop the exploration. We would, however, like to explore (randomly if necessary).
            # how to accomplish that?
            #if using suggestions in comments on line 33, replace 0 with n
            if ok:
                a = policy_bank.get_best_action(0, u1, s1_features.reshape((1,num_features)))
            else:
                pr = np.zeros([4,1])
                pr_sum = 0
                pr_select = np.zeros([5,1])
                for a in actions:
                    pr_sum += np.exp(q[s][u1_true][a])
                for a in actions:
                    pr[a] = np.exp(q[s][u1_true][a])/pr_sum
                pr_select[0] = 0
                pr_select[1] = pr[0]
                pr_select[2] = pr[0]+pr[1]
                pr_select[3] = pr[0]+pr[1]+pr[2]
                pr_select[4] = 1

                randn = random.random()
                a_selected = -1
                for a in actions:
                    if randn >= pr_select[a] and randn <= pr_select[a+1]:
                        a_selected = a
                        break

                a = a_selected

        # updating the curriculum
        curriculum.add_step()

        # Executing the action
        if tester_learned.game_type=="trafficworld":
            events = task.get_true_propositions_action(a)
            task.execute_action(a)
            a = task.get_last_action() # due to MDP slip
        else:
            task.execute_action(a)
            a = task.get_last_action() # due to MDP slip
            events = task.get_true_propositions()
        s2, s2_features = task.get_state_and_features()
        s_new = np.where(s2_features==1)[0][0]

        u2 = rm_learned.get_next_state(u1, events)
        u2_true = rm_true.get_next_state(u1_true,events)
        reward = rm_true.get_reward(u1_true,u2_true,s1,a,s2)

     #   q[s][u1_true][a] = (1 - alpha) * q[s][u1_true][a] + alpha * (reward + gamma * np.amax(q[s_new][u2_true]))

        sy = s%9
        sx = (s-sy)/9
        synew = s_new % 9
        sxnew = (s_new - synew) / 9
        a1=a

        if (events == "f"):
            events

        all_events.append(events)

        if reward>0:
            reward

        training_reward += reward

        # Getting rewards and next states for each reward machine
        rewards, next_states = [],[]
        rewards_hyp, next_states_hyp = [],[]
        j_rewards, j_next_states = rm_true.get_rewards_and_next_states(s1, a, s2, events)
        rewards.append(j_rewards)
        next_states.append(j_next_states)

        j_rewards_hyp, j_next_states_hyp = rm_learned.get_rewards_and_next_states(s1, a, s2, events)
        rewards_hyp.append(j_rewards_hyp)
        next_states_hyp.append(j_next_states_hyp)

        # Mapping rewards and next states to specific policies in the policy bank

        rewards_hyp = policy_bank.select_rewards(rewards_hyp)
        next_policies = policy_bank.select_next_policies(next_states_hyp)

        # Adding this experience to the experience replay buffer
        replay_buffer.add(s1_features, a, s2_features, rewards_hyp, next_policies)

        # Learning
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.train_freq == 0:
            if learning_params.prioritized_replay:
                experience = replay_buffer.sample(learning_params.batch_size, beta=beta_schedule.value(curriculum.get_current_step()))
                S1, A, S2, Rs, NPs, weights, batch_idxes = experience
            else:
                S1, A, S2, Rs, NPs = replay_buffer.sample(learning_params.batch_size)
                weights, batch_idxes = None, None
            abs_td_errors = policy_bank.learn(S1, A, S2, Rs, NPs, weights, has_been) # returns the absolute td_error
            if learning_params.prioritized_replay:
                new_priorities = abs_td_errors + learning_params.prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)
            
        # Updating the target network
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.target_network_update_freq == 0:
            policy_bank.update_target_network()

        # Printing
        if show_print and (t+1) % learning_params.print_freq == 0:
            print("Step:", t+1, "\tTotal reward:", training_reward)

        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq==0:
            testing_reward = tester_learned.run_test(curriculum.get_current_step(), sess, run_aqrm_test, rm_learned, rm_true, is_rm_learned, q, policy_bank, num_features)


        if is_rm_learned==0:
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_true):
                # Restarting the game
                task = Game(task_params)
                if curriculum.stop_task(t):
                    break
                s2, s2_features = task.get_state_and_features()
                u2_true = rm_true.get_initial_state()


        else:
            if task.is_env_game_over() or rm_learned.is_terminal_state(u2) or rm_true.is_terminal_state(u2_true):
                # Restarting the game
                task = Game(task_params)

                if curriculum.stop_task(t):
                    break

                s2, s2_features = task.get_state_and_features()
                u2_true = rm_true.get_initial_state()
                u2 = rm_learned.get_initial_state()


        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2
        u1_true = u2_true

    if rm_true.is_terminal_state(u2_true):
        checker = rm_learned.is_terminal_state(u2)

    if (is_rm_learned) and (not rm_learned.is_terminal_state(u2)) and (not rm_true.is_terminal_state(u2_true)):
        is_conflicting = 0
    elif is_rm_learned and (rm_learned.is_terminal_state(u2) and rm_true.is_terminal_state(u2_true)):
        is_conflicting = 0
    else:
        is_conflicting = 1

    step_count=t

    if testing_reward is None:
        is_test_result = 0
        testing_reward = previous_testing_reward
    else:
        is_test_result = 1


    if show_print: print("Done! Total reward:", training_reward)

    return all_events, training_reward, step_count, is_conflicting, testing_reward, is_test_result, q


def run_aqrm_test(sess, reward_machines, task_params, rm, rm_true, is_learned, q, learning_params, testing_params, optimal, policy_bank, num_features):
    # Initializing parameters
    task = Game(task_params)
    s1, s1_features = task.get_state_and_features()

    u1 = rm.get_initial_state()
    u1_true = rm_true.get_initial_state()

    alpha = 0.8
    gamma = 0.99
    w = 0
    ok = 1
    has_been = [0,0]

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Choosing an action to perform

        actions = task.get_actions()
        s = np.where(s1_features==1)[0][0]
        if ok:
            a = policy_bank.get_best_action(0, u1, s1_features.reshape((1,num_features)))
        else:
            pr = np.zeros([4,1])
            pr_sum = 0
            pr_select = np.zeros([5,1])
            for a in actions:
                pr_sum += np.exp(q[s][u1_true][a])
            for a in actions:
                pr[a] = np.exp(q[s][u1_true][a])/pr_sum
            pr_select[0] = 0
            pr_select[1] = pr[0]
            pr_select[2] = pr[0]+pr[1]
            pr_select[3] = pr[0]+pr[1]+pr[2]
            pr_select[4] = 1


            randn = random.random()
            a_selected = -1
            for a in actions:
                if randn >= pr_select[a] and randn <= pr_select[a+1]:
                    a_selected = a
                    break

            a = a_selected

        # Executing the action
        if task_params.game_type=="trafficworld":
            event = task.get_true_propositions_action(a)
            task.execute_action(a)
            a = task.get_last_action() # due to MDP slip
        else:
            task.execute_action(a)
            a = task.get_last_action() # due to MDP slip
            event = task.get_true_propositions()
        s2, s2_features = task.get_state_and_features()
        u2 = rm.get_next_state(u1, event)
        u2_true = rm_true.get_next_state(u1_true, event)
        r = rm_true.get_reward(u1_true,u2_true,s1,a,s2)
        s_new = np.where(s2_features==1)[0][0]

        #  q[s][u1_true][a] = (1 - alpha) * q[s][u1_true][a] + alpha * (r + gamma * np.amax(q[s_new][u2_true]))

        if (event == "f"):
            event

        r_total += r * learning_params.gamma**t # used in original graphing framework
        
        # Restarting the environment (Game Over)
        if is_learned==0:
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_true):
                break

        else:
            if task.is_env_game_over() or rm.is_terminal_state(u2):
                break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2
        u1_true = u2_true

    if rm_true.is_terminal_state(u2_true) and r>0:
        return 1
    else:
        return 0

    return r_total

def _remove_files_from_folder(relative_path):

    dirname = os.path.abspath(os.path.dirname(__file__))


    parent_folder = os.path.normpath(os.path.join(dirname, relative_path))

    if os.path.isdir(parent_folder):
        for filename in os.listdir(parent_folder):
            absPath = os.path.join(parent_folder, filename)
            subprocess.run(["rm", absPath])
    else:
        print("There is no directory {}".format(parent_folder))


def run_aqrm_experiments(alg_name, tester, tester_learned, curriculum, num_times, show_print, show_plots, is_SAT):
    learning_params = tester_learned.learning_params
    testing_params = tester_learned.testing_params

    # just in case, delete all temporary files
    dirname = os.path.abspath(os.path.dirname(__file__))
    _remove_files_from_folder("../automata_learning_utils/data")

    # Running the tasks 'num_times'
    time_init = time.time()
    plot_dict = dict()
    rewards_plot = list()

    new_traces = Traces(set(), set())

    for t in range(num_times):
        # Setting the random seed to 't'

        random.seed(t)
        sess = tf.Session()
        open('./automata_learning_utils/data/data.txt','w').close
        open('./automata_learning_utils/data/automaton.txt','w').close


        # Reseting default values
        curriculum.restart()

        # Creating the experience replay buffer
        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size, learning_params.prioritized_replay, learning_params.prioritized_replay_alpha, learning_params.prioritized_replay_beta0, curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)
        hm_file = './automata_learning/hypothesis_machine.txt'
        shutil.copy(hm_file,'./automata_learning_utils/data/rm.txt') #######

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())
        q = np.zeros([1681,8,4])

        num_episodes = 0
        total = 0
        learned = 0
        step = 0
        enter_loop = 1
        num_conflicting_since_learn = 0
        update_rm = 0
        refreshed = 0
        testing_step = 0

        hypothesis_machine = tester.get_hypothesis_machine()
        tester_learned.update_hypothesis_machine()
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, hypothesis_machine)

        # Task loop
        automata_history = []
        rewards = list()
        episodes = list()
        steps = list()
        testing_reward = 0 #initializes value
        all_traces = Traces(set(),set())
        epsilon = 0.3
        tt=t+1
        print("run index:", +tt)

        while not curriculum.stop_learning():
            num_episodes += 1

            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file_truth = '../experiments/craft/reward_machines/t1.txt' #set file path at beginning
            ###### is ^ redundant?
            # update hypothesis machine for every iteration of run_aqrm_task (in using two experiment files)
            # Running 'task_rm_id' for one episode
            hm_file_update =  './automata_learning_utils/data/rm.txt'

            if learned==0:
                rm_file_learned = hm_file
                if update_rm:
                    tf.reset_default_graph()
                    sess = tf.Session()
                    policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, hypothesis_machine)
                    update_rm = 0
                    refreshed = 1
                    tester_learned.update_hypothesis_machine_file('./automata_learning/hypothesis_machine.txt')
                    tester_learned.update_hypothesis_machine()
                    all_traces = Traces(set(),set())
                    num_conflicting_since_learn = 0
                  #  q = np.zeros([1681,4,4])
                    enter_loop = 1
            elif update_rm:
                rm_file_learned = hm_file_update
                tf.reset_default_graph()
                sess = tf.Session()

                # Reseting default values
                #####curriculum.restart()

                # Creating the experience replay buffer
                replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
                                                                               learning_params.prioritized_replay,
                                                                               learning_params.prioritized_replay_alpha,
                                                                               learning_params.prioritized_replay_beta0,
                                                                               curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

                # Creating policy bank
                task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
                num_features = len(task_aux.get_features())
                num_actions = len(task_aux.get_actions())
                rm_learned = tester_learned.get_hypothesis_machine() # used to be rm_learned = tester_learned.get_reward_machines()[0]
                if len(rm_learned.U)<=15:
                    print("number of states:" + str(len(rm_learned.U)))
                    #policy_bank = qrm.policy_bank_dqn.PolicyBankDQN(sess, num_actions, num_features, learning_params, tester_current.get_reward_machines())
                    policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, rm_learned)
                else:
                    print("number of states:" + str(len(rm_learned.U)))
                    tf.reset_default_graph()
                    sess = tf.Session()
                    policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, hypothesis_machine)
                    update_rm = 0
                    refreshed = 1
                    tester_learned.update_hypothesis_machine_file('./automata_learning/hypothesis_machine.txt')
                    tester_learned.update_hypothesis_machine()
                    all_traces = Traces(set(), set())
                    num_conflicting_since_learn = 0
                    q = np.zeros([1681, 4, 4])
                    enter_loop = 1
                    learned = 0

                update_rm = 0

            else:
                pass
            automata_history.append(rm_file_learned) #####fix this

            epsilon = epsilon*0.99

            all_events, found_reward, stepcount, conflicting, testing_reward, is_test, q = run_aqrm_task(sess, epsilon, rm_file_truth, rm_file_learned, policy_bank, tester, tester_learned, curriculum, replay_buffer, beta_schedule, show_print, learned, step, testing_reward, q)




            #set up traces; we remove anything foreign to our ground truth formula

            if tester.game_type=="officeworld":
                while 'h' in all_events:
                   all_events.remove('h')
            elif tester.game_type=="trafficworld":
                while 'f' in all_events:
                   all_events.remove('f')
                while 'g' in all_events:
                   all_events.remove('g')
            elif tester.game_type=="craftworld":
                while 'd' in all_events:
                   all_events.remove('d')
                while 'g' in all_events:
                   all_events.remove('g')
                while 'h' in all_events:
                   all_events.remove('h')
 
            while '' in all_events:
                all_events.remove('')
            if (conflicting==1 or refreshed==1):
                all_traces.add_trace(all_events, found_reward, learned)

            if (num_episodes%100==0):
                print("run index:", +tt)
                toprint = "Total training reward at "+str(step)+": "+str(total)
                print(toprint)

            if num_episodes>5000:
                num_episodes

            total += found_reward
            step += stepcount
            num_conflicting_since_learn += conflicting
            rewards.append(found_reward)
            episodes.append(num_episodes)
            steps.append(step)

            if is_test:
                testing_step += testing_params.test_freq
                if testing_step in plot_dict:
                    plot_dict[testing_step].append(testing_reward)
                else:
                    plot_dict[testing_step] = [testing_reward]


            if learned==1:
                # if len(rm_learned.U)>learning_params.relearn_period:
                #     learned = 0
                #     update_rm = 1

                if num_episodes%learning_params.relearn_period==0 and (num_conflicting_since_learn>0):
                    enter_loop = 1

                if conflicting==1:
                    new_traces.add_trace(all_events, found_reward, learned)


            if (len(all_traces.positive)>=learning_params.enter_loop) and enter_loop:

                positive = set()
                negative = set()

                if learned==0:
                    if len(all_traces.positive)>0:
                         for i in list(all_traces.positive):
                             if all_traces.symbol_to_trace(i) not in positive:
                                 positive.add(all_traces.symbol_to_trace(i))
                    if len(all_traces.negative)>0:
                        for i in list(all_traces.negative):
                            if all_traces.symbol_to_trace(i) not in negative:
                                negative.add(all_traces.symbol_to_trace(i))
                else:
                     if len(new_traces.positive)>0:
                        for i in list(new_traces.positive):
                             if new_traces.symbol_to_trace(i) not in positive:
                                 positive.add(new_traces.symbol_to_trace(i))
                     if len(new_traces.negative)>0 and len(all_traces.negative):
                         for i in list(new_traces.negative):
                             if new_traces.symbol_to_trace(i) not in negative:
                                 negative.add(new_traces.symbol_to_trace(i))


                positive_new = set() ## to get rid of redundant prefixes
                negative_new = set()

                if not learned:
                    for ptrace in positive:
                        new_trace = list()
                        previous_prefix = 100 #arbitrary
                        for prefix in ptrace:
                            if prefix != previous_prefix:
                                new_trace.append(prefix)
                            previous_prefix = prefix
                        positive_new.add(tuple(new_trace))

                    for ntrace in negative:
                        new_trace = list()
                        previous_prefix = 100 #arbitrary
                        for prefix in ntrace:
                            if prefix != previous_prefix:
                                new_trace.append(prefix)
                            previous_prefix = prefix
                        negative_new.add(tuple(new_trace))
                    if tester.game_type=="trafficworld":
                        negative_to_store = set(random.sample(negative_new, 50))
                    else:
                        negative_to_store = negative_new
                    positive_to_store = positive_new
                    negative_new = negative_to_store
                    positive_new = positive_to_store

                    negative = set()
                    positive = set()

                else:
                    for ptrace in positive:
                        new_trace = list()
                        for prefix in ptrace:
                            new_trace.append(prefix)
                        positive_to_store.add(tuple(new_trace))
                        positive_new = positive_to_store
                        negative_new = negative_to_store

                    for ntrace in negative:
                        new_trace = list()
                        for prefix in ntrace:
                            new_trace.append(prefix)
                        negative_to_store.add(tuple(new_trace))
                        positive_new = positive_to_store
                        negative_new = negative_to_store

                traces_numerical = Traces(positive_new, negative_new)
                traces_file = './automata_learning_utils/data/data.txt'
                traces_numerical.export_traces(traces_file)

                if learned == 1:
                    shutil.copy('./automata_learning_utils/data/rm.txt', '../experiments/craft/use_past/t2.txt')
                # else:
                #     shutil.copy('./automata_learning/hypothesis_machine.txt', '../experiments/craft/use_past/t2.txt')

                automaton_visualization_filename = al_utils.learn_automaton(traces_file,show_plots,is_SAT)
                if show_plots:
                    subprocess.run(["xdot",automaton_visualization_filename])

              # t2 is previous, t1 is new
                hm_file_update = './automata_learning_utils/data/rm.txt'
                # if learned==1:
                #     shutil.copy(hm_file_update, '../experiments/craft/use_past/t2.txt')
                # else:
                #     shutil.copy('./automata_learning/hypothesis_machine.txt', '../experiments/craft/use_past/t2.txt')

                all_traces.rm_trace_to_symbol(hm_file_update)
                all_traces.fix_rmfiles(hm_file_update)

                if learned == 0:
                    shutil.copy('./automata_learning_utils/data/rm.txt',
                                             '../experiments/craft/use_past/t2.txt')


                # the learning should happen here
                tester_learned.update_hypothesis_machine_file(hm_file_update) ## NOTE WHICH TESTER IS USED
                tester_learned.update_hypothesis_machine()

                print("learning")
                parent_path = os.path.abspath("../experiments/craft/use_past/")
                os.makedirs(parent_path, exist_ok=True)
                shutil.copy(hm_file_update, '../experiments/craft/use_past/t1.txt')
                current_and_previous_rms = '../experiments/craft/tests/use_previous_experience.txt'

                tester_current = Tester(learning_params,testing_params,current_and_previous_rms)

                learned = 1
                enter_loop = 0
                num_conflicting_since_learn = 0
                update_rm = 1

                #if (len(all_traces.positive)+len(all_traces.negative))>100:
                #    all_traces = Traces(set(),set())

            if num_episodes%learning_params.relearn_period==0:
                new_traces = Traces(set(), set())

        tf.reset_default_graph()
        sess.close()

        # Backing up the results
        print('Finished iteration ',t)

    # Showing results

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()


    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps_plot = list()

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        rewards_plot.append(sum(plot_dict[step])/len(plot_dict[step]))
        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps_plot.append(step)

    time_elapsed = time.clock() - time_start

    tester.plot_performance(steps_plot,prc_25,prc_50,prc_75)
    tester.plot_this(steps_plot,rewards_plot)

    if alg_name=="jirp":
        if is_SAT==1:
            algorithm_name = "jirpsat"
        else:
            algorithm_name = "jirprpni"
    else:
        algorithm_name = alg_name

    for character in tester.world.tasks[0]:
        if str.isdigit(character):
            task_id = character
            filename = ("../plotdata/") + (tester.game_type) + ("") + (task_id) + ("") + (
                algorithm_name) + ".csv"

    with open(filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(list(plot_dict.values()))


    avg_filename = ("../plotdata/") + ("avgreward_") + (tester.game_type) + ("") + (task_id) + ("") + (
                algorithm_name) + ".txt"

    with open(avg_filename, 'w') as f:
        f.write("%s\n" % str(sum(rewards_plot) / len(rewards_plot)))
        for item in rewards_plot:
            f.write("%s\n" % item)

    print('')
    print('')
    print('')
    print('Time taken:')
    print(time_elapsed)
