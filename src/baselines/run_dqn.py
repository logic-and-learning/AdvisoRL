import random, time
import csv
import numpy as np
import tensorflow as tf
from worlds.game import *
from tester.saver import Saver
from common.schedules import LinearSchedule
from baselines.policy_bank import PolicyBank
from reward_machines.reward_machine import RewardMachine

def run_dqn_baseline(sess, rm_file, policy_bank, tester, curriculum, show_print, previous_test):
    """
    This code runs one training episode.
        - rm_file: It is the path towards the RM machine to solve on this episode
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    rm_id = tester.get_reward_machine_id_from_file(list(tester.file_to_reward_machine.keys())[0])
    task_params = tester.get_task_params(list(tester.file_to_reward_machine.keys())[0])
    task = Game(task_params)
    actions = task.get_actions()
    num_features = policy_bank.get_number_features(rm_id)
    num_steps = learning_params.max_timesteps_per_task
    rm = tester.get_hypothesis_machine()
    rm_true = RewardMachine(list(tester.file_to_reward_machine.keys())[0])
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    s1_location = np.where(s1_features)
    u1 = rm.get_initial_state()
    u1_true = rm_true.get_initial_state()
    feature_memory = np.zeros((1,learning_params.memory_size))
    feature_memory_0 = np.zeros((1,learning_params.memory_size))
    # feature_memory[-1][-1] = s1_location[0][0]
    is_test = 0
    # event_label = np.zeros((1,))



    checker = 0


    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)

    for t in range(num_steps):

        current_label = np.zeros((1,7))

        # Choosing an action to perform
        if random.random() < 0.3:
            a = random.choice(actions)
        else:
            a = policy_bank.get_best_action(rm_id, feature_memory, s1_location[0][0])
            if a != 0:
                a

        # updating the curriculum
        curriculum.add_step()
        policy_bank.add_step(rm_id)

        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        s2_location = np.where(s2_features)
        events = task.get_true_propositions()

        u2 = rm.get_next_state(u1, events)
        u2_true = rm_true.get_next_state(u1_true,events)
        reward = rm_true.get_reward(u1_true,u2_true,s1,a,s2)

        if reward>0:
            reward

        if events=="a":
            current_label[0][0] = 1

        elif events=="b":
            current_label[0][1] = 1

        elif events=="c":
            current_label[0][2] = 1

        elif events == "d":
            current_label[0][3] = 1

        elif events == "e":
            current_label[0][4] = 1

        elif events=="f":
            current_label[0][5] = 1

        elif events=="g":
            current_label[0][6] = 1

        done = task.is_env_game_over() or rm_true.is_terminal_state(u2_true)
        training_reward += reward

        feature_memory_0 = feature_memory
        if min(current_label[0]==[0,0,0,0,0,0,0])==False:
            label = np.where(current_label)[1][0]+1
        else:
            label = 0

        feature_memory = np.delete(feature_memory, 0)
        feature_memory = np.append(feature_memory, label)

        # Saving this transition
        policy_bank.add_experience(rm_id, feature_memory_0, s1_location[0][0], a, reward, feature_memory, s2_location[0][0], float(done))

        # Learning
        if policy_bank.get_step(rm_id) > learning_params.learning_starts and policy_bank.get_step(rm_id) % learning_params.train_freq == 0:
            policy_bank.learn(rm_id)

        # Updating the target network
        if policy_bank.get_step(rm_id) > learning_params.learning_starts and policy_bank.get_step(rm_id) % learning_params.target_network_update_freq == 0:
            policy_bank.update_target_network(rm_id)


        # Testing
        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
            # testing_reward = tester.run_test(curriculum.get_current_step(), sess, run_dqn_test, rm, RewardMachine(list(tester.file_to_reward_machine.keys())[0]), 0, 0, policy_bank, rm_id)
            is_test = 1
            # testing_reward = r

        if is_test and reward==1:
            testing_reward = tester.run_test(curriculum.get_current_step(), sess, run_dqn_test, rm, RewardMachine(list(tester.file_to_reward_machine.keys())[0]), 0, reward, policy_bank, rm_id)
            return t, 1, 1

        # Restarting the environment (Game Over)
        if done:
            # Restarting the game
            task = Game(task_params)
            s2, s2_features = task.get_state_and_features()
            u2 = rm.get_initial_state()
            u2_true = rm_true.get_initial_state()

            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1, s1_features, s1_location, u1, u1_true = s2, s2_features, s2_location, u2, u2_true

        # loop_size = len(feature_memory)

        # for i in range(0,loop_size):
        #
        #     if i<num_features:
        #         np.delete(feature_memory,0)
        #
        #
        # for i in s1_features:
        #     np.append(feature_memory,i)

        step_count = t

        if is_test == 0:
            is_test_result = 0
            testing_reward = previous_test
        else:
            is_test_result = 1

    if is_test:
        testing_reward = tester.run_test(curriculum.get_current_step(), sess, run_dqn_test, rm,RewardMachine(list(tester.file_to_reward_machine.keys())[0]), 0, 0,policy_bank, rm_id)
        return step_count, 0, 1

    return step_count, testing_reward, is_test_result

def run_dqn_test(sess, reward_machines, task_params, rm_learned, rm_true, is_learned, q, learning_params, testing_params, discard,policy_bank, rm_id):
    return q

# def run_dqn_test(sess, reward_machines, task_params, rm_learned, rm_true, is_learned, q, learning_params, testing_params, discard,policy_bank, rm_id):
#     # Initializing parameters
#     task = Game(task_params)
#     rm = rm_learned
#     s1, s1_features = task.get_state_and_features()
#     s1_location = np.where(s1_features)
#     u1 = rm.get_initial_state()
#     u1_true = rm_true.get_initial_state()
#     num_features = policy_bank.get_number_features(rm_id)
#     feature_memory = np.zeros((1,learning_params.memory_size))
#
#
#
#     # Starting interaction with the environment
#     r_total = 0
#     for t in range(testing_params.num_steps):
#         current_label = np.zeros((1, 7))
#         # Choosing an action to perform
#         if random.random()<0.3:
#             a = random.choice(task.get_actions())
#         else:
#             a = policy_bank.get_best_action(rm_id, feature_memory, s1_location[0][0])
#
#         # Executing the action
#         task.execute_action(a)
#         s2, s2_features = task.get_state_and_features()
#         s2_location = np.where(s2_features)
#         u2 = rm.get_next_state(u1, task.get_true_propositions())
#         events = task.get_true_propositions()
#         u2_true = rm_true.get_next_state(u1_true,task.get_true_propositions())
#         r = rm_true.get_reward(u1_true,u2_true,s1,a,s2)
#
#         if r>0:
#             r
#
#         if events=="a":
#             current_label[0][0] = 1
#
#         elif events=="b":
#             current_label[0][1] = 1
#
#         elif events=="c":
#             current_label[0][2] = 1
#
#         elif events == "d":
#             current_label[0][3] = 1
#
#         elif events == "e":
#             current_label[0][4] = 1
#
#         elif events=="f":
#             current_label[0][5] = 1
#
#         elif events=="g":
#             current_label[0][6] = 1
#
#
#         if min(current_label[0]==[0,0,0,0,0,0,0])==False:
#             label = np.where(current_label)[1][0]+1
#         else:
#             label = 0
#
#         r_total += r * learning_params.gamma**t
#
#         # Restarting the environment (Game Over)
#         if task.is_env_game_over() or rm_true.is_terminal_state(u2_true):
#             break
#
#         # Moving to the next state
#         s1, s1_features, s1_location, u1 = s2, s2_features, s2_location, u2
#         u1_true = u2_true
#
#
#         # for i in range(0,loop_size):
#         #
#         #     if i<num_features:
#         #         np.delete(feature_memory,0)
#         #
#         #
#         # for i in s1_features:
#         #     np.append(feature_memory,i)
#         feature_memory = np.delete(feature_memory, 0)
#         feature_memory = np.append(feature_memory, label)
#     #
#     # if rm_true.is_terminal_state(u2_true):
#     #     return 1
#     # else:
#     #     return 0
#
#     # return r_total
#     return r


def run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print):
    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    step = 0
    steps = list()
    rewards = list()
    plot_dict = dict()

    if isinstance(num_times, int):
        num_times = range(num_times)
    elif isinstance(num_times, tuple):
        num_times = range(*num_times)
    for t_i,t in enumerate(num_times):
        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()
        previous_test = 0
        testing_step = 0

        # Reseting default values
        curriculum.restart()

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())

        # policy_bank = PolicyBank(sess, num_actions, num_features, learning_params, curriculum, [tester.get_hypothesis_machine()])
        policy_bank = PolicyBank(sess, num_actions, num_features, learning_params, curriculum, [tester.get_hypothesis_machine()])

        # Task loop
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file = curriculum.get_next_task()
            step_count, testing_reward, is_test = run_dqn_baseline(sess, rm_file, policy_bank, tester, curriculum, show_print, previous_test)
            step += step_count
            steps.append(step)

            if is_test:
                testing_step += tester.testing_params.test_freq
                if testing_step in plot_dict:
                    plot_dict[testing_step].append(testing_reward)
                else:
                    plot_dict[testing_step] = [testing_reward]


        tf.reset_default_graph()
        sess.close()

        # Backing up the results
        saver.save_results()

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()
    rewards_plot = list()
    steps_plot = list()


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


    tester.plot_performance(steps_plot,prc_25,prc_50,prc_75)
    tester.plot_this(steps_plot,rewards_plot)

    for character in tester.world.tasks[0]:
        if str.isdigit(character):
            task_id = character
            filename = ("../plotdata/") + (tester.game_type) + ("") + (task_id) + ("") + (
                alg_name) + ".csv"

    with open(filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(list(plot_dict.values()))


    avg_filename = ("../plotdata/") + ("avgreward_") + (tester.game_type) + ("") + (task_id) + ("") + (
                alg_name) + ".txt"

    with open(avg_filename, 'w') as f:
        f.write("%s\n" % str(sum(rewards_plot) / len(rewards_plot)))
        for item in rewards_plot:
            f.write("%s\n" % item)


    # Showing results
    tester.show_results()
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")
