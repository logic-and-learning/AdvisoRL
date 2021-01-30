import numpy as np
import random, time
import tensorflow as tf
from worlds.game import *
from hrl.policy_bank_dqn import PolicyBankDQN
from common.schedules import LinearSchedule
from common.replay_buffer import create_experience_replay_buffer
from tester.saver import Saver
from os import listdir
from os.path import isfile, join
from reward_machines.reward_machine import RewardMachine
from baselines.hrl import MetaController
import csv

def run_hrl_baseline(sess, q, rm_file, meta_controllers, options, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print, current_step, previous_test):
    """
    Strategy:
        - I'll learn a tabular metacontroller over the posible subpolicies
        - Initialice a regular policy bank with eventually subpolicies (e.g. Fa, Fb, Fc, Fb, Fd)
        - Learn as usual
        - Pick actions using the sequence
    """

    # Initializing parameters
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    reward_machines = tester.get_reward_machines()
    rm_id = tester.get_reward_machine_id_from_file(rm_file)
    task_params = tester.get_task_params(rm_file)
    task = Game(task_params)
    actions = task.get_actions()
    num_features = len(task.get_features())
    meta_controller = meta_controllers[rm_id]
    rm = reward_machines[rm_id]
    num_steps = learning_params.max_timesteps_per_task
    training_reward = 0
    testing_reward = 0
    is_test = 0

    g = 1
    N = 20 #episodes per q update
    alpha = 0.8
    gamma = 0.99
    horizon_reward = 0
    mc_u1 = 0
    u = 0
    mc_a  =0
    s=0
    s_new=0
    reward = 0
    all_events = list()


    # Starting interaction with the environment
    if show_print: print("Executing", num_steps, "actions...")
    t = 0
    curriculum_stop = False

    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()
    
    while t < learning_params.max_timesteps_per_task and not curriculum_stop:
        # selecting a macro action from the meta controller
        mc_s1, mc_s1_features, mc_u1 = s1, s1_features, u1
        mc_r  = []

        T = 1

        u_pool = range(0,learning_params.pool_size)
        pr = np.zeros([learning_params.pool_size,1])
        pr_sum = 0
        pr_select = np.zeros([learning_params.pool_size+1,1])
        for u_ in u_pool:
            pr_sum += np.exp(q[s][u_]*T)
        for u_ in u_pool:
            pr[u_] = np.exp(q[s][u_]*T)/pr_sum
        for index in range(1,learning_params.pool_size):
            for pr_pos in range(0,index):
                pr_select[index] += pr[pr_pos]
        pr_select[-1] = 1


        randn = random.random()
        u_selected = -1
        for u_ in u_pool:
            if randn >= pr_select[u_][0] and randn <= pr_select[u_+1][0]:
                u_selected = u_
                break

        u_new = u_selected

        if reward>0:
            testy = 0

        q[s][u] = (1-alpha)*q[s][u] + alpha*(reward + gamma*np.amax(q[s_new][u_new]))


        mc_a = u_new
        u = u_new

        if t%N==0:
            horizon_reward = 0
        else:
            horizon_reward += reward

        mc_option = meta_controller.get_option(mc_a)  # tuple <rm_id,u_0>

        mc_done = False
        if show_print: print(mc_option)

        # The selected option must be executed at least one step (i.e. len(mc_r) == 0)
        #while len(mc_r) == 0:
        # or not meta_controller.finish_option(mc_a, task.get_true_propositions()):
        current_step += 1

            # Choosing an action to perform
        if random.random() < 0.15: 
            a = random.choice(actions)
        else: 
            a = policy_bank.get_best_action(mc_option[0], mc_option[1], s1_features.reshape((1,num_features)))

            # updating the curriculum
        curriculum.add_step()
            
            # Executing the action
        if tester.game_type=="trafficworld":
            events = task.get_true_propositions_action(a)
            task.execute_action(a)
            a = task.get_last_action() # due to MDP slip
        else:
            task.execute_action(a)
            a = task.get_last_action() # due to MDP slip
            events = task.get_true_propositions()
        s2, s2_features = task.get_state_and_features()
        all_events.append(events)
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1,u2,s1,a,s2)
        training_reward += reward
        s = np.where(s1_features==1)[0][0]
        s_new = np.where(s2_features==1)[0][0]

        sy = s%11+1
        sx = (s-sy+1)/11+1
        synew = s_new % 11+1
        sxnew = (s_new - synew+1) / 11+1
        a1=a

        if reward>0:
            reward


        # updating the reward for the meta controller
        mc_r.append(reward)

        # Getting rewards and next states for each option
        rewards, next_states = [],[]
        for j in range(len(options)):
            j_rewards, j_next_states = options[j].get_rewards_and_next_states(s1, a, s2, events)
            rewards.append(j_rewards)
            next_states.append(j_next_states)
        # Mapping rewards and next states to specific policies in the policy bank
        rewards = policy_bank.select_rewards(rewards)
        next_policies = policy_bank.select_next_policies(next_states)

        # Adding this experience to the experience replay buffer
        replay_buffer.add(s1_features, a, s2_features, rewards, next_policies)

        # Learning
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.train_freq == 0:
            if learning_params.prioritized_replay:
                experience = replay_buffer.sample(learning_params.batch_size, beta=beta_schedule.value(curriculum.get_current_step()))
                S1, A, S2, Rs, NPs, weights, batch_idxes = experience
            else:
                S1, A, S2, Rs, NPs = replay_buffer.sample(learning_params.batch_size)
                weights, batch_idxes = None, None
            abs_td_errors = policy_bank.learn(S1, A, S2, Rs, NPs, weights) # returns the absolute td_error
            if learning_params.prioritized_replay:
                new_priorities = abs_td_errors + learning_params.prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            # Updating the target network
        if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.target_network_update_freq == 0:
            policy_bank.update_target_network()

            # Printing
        if show_print and (t+1) % learning_params.print_freq == 0:
            print("Step:", t+1, "\tTotal reward:", training_reward)

        # Testing
        if tester.testing_params.test and curriculum.get_current_step() % tester.testing_params.test_freq == 0:
            testing_reward, q = tester.run_test(curriculum.get_current_step(), sess, q, run_hrl_baseline_test, meta_controllers, policy_bank, num_features)
            is_test = 1

            # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2):
                # Restarting the game
            task = Game(task_params)
            s2, s2_features = task.get_state_and_features()
            u2 = rm.get_initial_state()

            mc_done = True

            if curriculum.stop_task(t):
                curriculum_stop = True

            # checking the steps time-out
        if curriculum.stop_learning():
            curriculum_stop = True

            # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

        t += 1
        if t == learning_params.max_timesteps_per_task or curriculum_stop or mc_done: 
            break


        # learning on the meta controller
        mc_s2, mc_s2_features, mc_u2 = s1, s1_features, u
        mc_reward = _get_discounted_reward(mc_r, learning_params.gamma)
        mc_steps = len(mc_r)

        #meta_controller.learn(mc_s1_features, mc_u1, mc_a, mc_reward, mc_s2_features, mc_u2, mc_done, mc_steps)

    #meta_controller.show()
    #input()
    step_count = t

    if is_test==0:
        is_test_result = 0
        testing_reward = previous_test
    else:
        is_test_result = 1


    return training_reward, step_count, testing_reward, is_test_result, q

def _get_discounted_reward(r_all, gamma):
    dictounted_r = 0
    for r in r_all[::-1]:
        dictounted_r = r + gamma*dictounted_r
    return dictounted_r

def run_hrl_baseline_test(sess, q, reward_machines, task_params, rm_id, learning_params, testing_params, meta_controllers, policy_bank, num_features):

    # Initializing parameters
    meta_controller = meta_controllers[rm_id]
    task = Game(task_params)
    rm = reward_machines[rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()
    horizon_reward = 0
    reward = 0


    # Starting interaction with the environment
    r_total = 0
    t = 0
    N = 20
    alpha = 0.8
    gamma = 0.99
    u = 0
    s=0
    s_new=0
    u_pool = range(0, learning_params.pool_size)

    while t < testing_params.num_steps:
        # selecting a macro action from the meta controller
        mc_s1, mc_s1_features, mc_u1 = s1, s1_features, u
        T = 1

        if random.random()<0.1:
            mc_a = random.choice(u_pool)
        else:
            pr = np.zeros([learning_params.pool_size, 1])
            pr_sum = 0
            pr_select = np.zeros([learning_params.pool_size + 1, 1])
            for u_ in u_pool:
                pr_sum += np.exp(q[s][u_]*T)
            for u_ in u_pool:
                pr[u_] = np.exp(q[s][u_]*T) / pr_sum
            for index in range(1, learning_params.pool_size):
                for pr_pos in range(0, index):
                    pr_select[index] += pr[pr_pos]
            pr_select[-1] = 1


            randn = random.random()
            u_selected = -1
            for u_ in u_pool:
                if randn >= pr_select[u_][0] and randn <= pr_select[u_+1][0]:
                    u_selected = u_
                    break

            u_new = u_selected


            q[s][u] = (1-alpha)*q[s][u] + alpha*(reward + gamma*np.amax(q[s_new][u_new]))

            mc_a = u_new


        mc_option = meta_controller.get_option(mc_a) # tuple <rm_id,u_0>

        # The selected option must be executed at least one step
        first = True

        # Choosing an action to perform
        a = policy_bank.get_best_action(mc_option[0], mc_option[1], s1_features.reshape((1,num_features)))
            
            # Executing the action
        task.execute_action(a)
        a = task.get_last_action() # due to MDP slip
        s2, s2_features = task.get_state_and_features()
        events = task.get_true_propositions()
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1,u2,s1,a,s2)
        r_total += reward * learning_params.gamma**t
        s = np.where(s1_features==1)[0][0]
        s_new = np.where(s2_features==1)[0][0]


            # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

        t += 1
            # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2) or t == testing_params.num_steps:
            break

    if rm.is_terminal_state(u2):
        return 1, q
    else:
        return 0, q

    return r_total

def _get_option_files(folder):
    return [f.replace(".txt","") for f in listdir(folder) if isfile(join(folder, f))]

def run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm):
    """
        NOTE: To implement this baseline, we encode each option as a reward machine with one transition
        - use_rm: Indicates whether to prune options using the reward machine
    """

    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    step = 0

    steps = list()
    rewards = list()
    plot_dict = dict()
    for t in range(num_times):
        tt=t+1
        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()
        testing_reward = 0
        testing_step = 0
        num_episodes = 0
        total = 0

        q = np.zeros([1681,8]) #second dimension is number of options


        # Reseting default values
        curriculum.restart()

        # Creating the experience replay buffer
        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size, learning_params.prioritized_replay, learning_params.prioritized_replay_alpha, learning_params.prioritized_replay_beta0, curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)      
        
        # Loading options for this experiment
        option_folder = "../experiments/%s/options/"%tester.get_world_name()

        options = [] # NOTE: The policy bank also uses this list (in the same order)
        option2file = []
        for option_file in _get_option_files(option_folder): # NOTE: The option id indicates what the option does (e.g. "a&!n")
            option = RewardMachine(join(option_folder, option_file + ".txt"))
            options.append(option)
            option2file.append(option_file)

        # getting num inputs and outputs net
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())

        # initializing the meta controllers (one metacontroller per task)
        meta_controllers = []
        reward_machines = tester.get_reward_machines()

        if tester.game_type == "trafficworld":
            options[0] = options[0]
            options[1] = options[6]
            options[2] = options[2]
            options[3] = options[3]
            options[4] = options[7]
            learning_params.pool_size = 5

        elif tester.game_type == "officeworld":
            options[0] = options[0]
            options[1] = options[6]
            options[2] = options[2]
            options[3] = options[3]
            options[4] = options[4]
            options[5] = options[5]
            learning_params.pool_size = 6

        else:
            options[0] = options[6]
            options[1] = options[7]
            options[2] = options[2]
            options[3] = options[3]
            learning_params.pool_size = 4


        for i in range(len(reward_machines)):
            rm = reward_machines[i]
            num_states = len(rm.get_states())
            policy_name = "Reward_Machine_%d"%i
            mc = MetaController(sess, policy_name, options, option2file, rm, use_rm, learning_params, num_features, num_states, show_print)
            meta_controllers.append(mc)

        # initializing the bank of policies with one policy per option
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, options)


        # Task loop
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file = curriculum.get_next_task()
            num_episodes += 1

            # Running 'rm_file' for one episode
            found_reward, step_count, testing_reward, is_test, q = run_hrl_baseline(sess, q, rm_file, meta_controllers, options, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print, step, testing_reward)
            step += step_count
            steps.append(step)
            rewards.append(found_reward)
            total += found_reward

            if (num_episodes%100==0):
                print("run index:", +tt)
                toprint = "Total training reward at "+str(step)+": "+str(total)
                print(toprint)

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

    # Showing results

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
