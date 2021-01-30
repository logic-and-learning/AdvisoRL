#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random, time, argparse, os.path, itertools
from automata_learning_with_policybank import aqrm
from automata_learning.aqrm import run_aqrm_experiments
from automata_learning.qlearning import run_qlearning_experiments
from baselines.run_dqn import run_dqn_experiments
from baselines.run_hrl import run_hrl_experiments
from tester.tester import Tester
from testerHRL.tester import TesterHRL
from tester_policybank.tester import TesterPolicyBank
from tester.tester_params import TestingParameters
from common.curriculum import CurriculumLearner
from qrm.learning_params import LearningParameters



def get_params_craft_world(experiment):
    step_unit = 400

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = step_unit
    testing_params.num_steps = step_unit  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.memory_size = 200
    learning_params.buffer_size = 1
    learning_params.relearn_period = 30
    learning_params.enter_loop = 10
    learning_params.lr = 1  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 50000
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 32
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 1000

    # Tabular case
    learning_params.tabular_case = False  # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 1500 * step_unit
    curriculum.min_steps = 1

    print("Water World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def get_params_office_world(experiment):
    # step_unit = 1000
    step_unit = 3000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq =  step_unit
    testing_params.num_steps = step_unit  # I'm giving one minute to the agent to solve the task
    # testing_params.num_steps = step_unit*2  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.memory_size = 200
    learning_params.buffer_size = 10
    learning_params.relearn_period = 30
    # learning_params.enter_loop = 10
    learning_params.enter_loop = 1
    learning_params.lr = 1e-4  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 1
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 10

    # Tabular case
    learning_params.tabular_case = False
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    # curriculum.total_steps = 400 * step_unit
    # curriculum.total_steps = 800 * step_unit
    curriculum.total_steps = 200 * step_unit
    curriculum.min_steps = 1

    print("Water World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def get_params_traffic_world(experiment):
    step_unit = 100

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.memory_size = 5
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.tabular_case = False
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.relearn_period = 100
    learning_params.enter_loop = 5
    learning_params.memory_size = 200
    learning_params.buffer_size = 1

    learning_params.lr = 1  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 10
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 1
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 10

    # These are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1
    learning_params.tabular_case = False  # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = False
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)


    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 20000*step_unit
    curriculum.min_steps = 1

    print("Traffic World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)


    return testing_params, learning_params, tester, curriculum



def get_params_taxi_world(experiment):
    # step_unit = 1000
    step_unit = 3000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq =  step_unit
    testing_params.num_steps = step_unit  # I'm giving one minute to the agent to solve the task
    # testing_params.num_steps = step_unit*2  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.memory_size = 200
    learning_params.buffer_size = 10
    learning_params.relearn_period = 30
    # learning_params.enter_loop = 10
    learning_params.enter_loop = 1
    learning_params.lr = 1e-4  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 1
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 10

    # Tabular case
    learning_params.tabular_case = False
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    # curriculum.total_steps = 400 * step_unit
    # curriculum.total_steps = 800 * step_unit
    curriculum.total_steps = 200 * step_unit
    curriculum.min_steps = 1

    print("Taxi World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def run_experiment(world, alg_name, experiment_known, experiment_learned, num_times, show_print, show_plots, al_alg_name, sat_alg_name, pysat_hints=None):
    if world == 'officeworld':
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_office_world(experiment_known)
        testing_params, learning_params, tester_l, curriculum = get_params_office_world(experiment_learned)
        ## allows for 2 sets of testers/curricula: one for previously known (ground truth) and one for learned info
    if world == 'craftworld':
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_craft_world(experiment_known)
        testing_params, learning_params, tester_l, curriculum = get_params_craft_world(experiment_learned)
    if world == 'trafficworld':
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_traffic_world(experiment_known)
        testing_params, learning_params, tester_l, curriculum = get_params_traffic_world(experiment_learned)
    if world == 'waterworld':
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_water_world(experiment_known)
        testing_params, learning_params, tester_l, curriculum = get_params_water_world(experiment_learned)
    if world == 'taxiworld':
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_taxi_world(experiment_known)
        testing_params, learning_params, tester_l, curriculum = get_params_taxi_world(experiment_learned)

    if alg_name == "ddqn":
        tester = TesterPolicyBank(learning_params, testing_params, experiment_known)
        run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print)

    if alg_name == "hrl":
        if world == 'craftworld':
            testing_params, learning_params, tester, curriculum = get_params_craft_world("../experiments/craft/tests/ground_truth.txt")
        elif world == 'officeworld':
            testing_params, learning_params, tester, curriculum = get_params_office_world("../experiments/office/tests/ground_truth.txt")
        tester = TesterHRL(learning_params, testing_params, experiment_known)
        run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm = False)

    #if (alg_name == "jirp") and (world== "trafficworld") and (not is_SAT):
    #    tester = TesterPolicyBank(learning_params, testing_params, experiment_known)
    #    tester_l = TesterPolicyBank(learning_params, testing_params, experiment_learned)
    #    aqrm.run_aqrm_experiments(alg_name, tester, tester_l, curriculum, num_times, show_print, show_plots, is_SAT)


    if alg_name == "jirp":
        run_aqrm_experiments(alg_name, tester, tester_l, curriculum, num_times, show_print, show_plots, al_alg_name, sat_alg_name, pysat_hints)

    if alg_name == "qlearning":
        run_qlearning_experiments(alg_name, tester, tester_l, curriculum, num_times, show_print, show_plots)


if __name__ == "__main__":

    # EXAMPLE: python3 run.py --algorithm="jirp" --world="craft" --map=0 --num_times=1 --show_plots=1 --al_alg_name=SAT

    # Getting params
    algorithms     = ["hrl", "jirp", "qlearning", "ddqn"]
    al_algorithms  = ["RPNI", "SAT", "PYSAT"]
    worlds         = ["office", "craft", "traffic", "taxi"]
    from automata_learning_utils.pysat import sat_algorithms

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='ddqn', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--world', default='traffic', type=str,
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--num_times', default='10', type=str,
                        help='It must be a number greater or equal to 1.') # TODO: better help
    parser.add_argument("--verbosity", help="increase output verbosity")
    parser.add_argument("--show_plots", default=0, help="1 for showing plots throughout the algorithm run, 0 otherwise")
    parser.add_argument("--al_algorithm", default=al_algorithms[0],
                        help='This parameter indicated which automaton learning algorithm to use with jirp. The options are: ' + str(al_algorithms))
    parser.add_argument("--sat_algorithm", default=sat_algorithms[0],
                        help='This parameter indicated which sat algorithm to use with jirp+pysat. The options are: ' + str(sat_algorithms))
    parser.add_argument("--pysat_hints", "--hints", action='append',
                        help='This parameter indicated hint(s) to use with jirppysat. Hints are substrings of the final word. Separate hints by ":" or ",".')


    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError("Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not(0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")
    # if args.num_times < 1: raise NotImplementedError("num_times must be greater than 0")

    # Running the experiment
    alg_name   = args.algorithm
    world      = args.world
    map_id     = args.map
    num_times  = tuple(int(n) for n in args.num_times.split(':'))
    show_print = args.verbosity is not None
    show_plots = (int(args.show_plots) == 1)
    al_alg_name = args.al_algorithm
    sat_alg_name = args.sat_algorithm
    if args.pysat_hints is None:
        pysat_hints = None
    else:
        pysat_hints = list(itertools.chain.from_iterable(
            hints.split(",")
            for hints in itertools.chain.from_iterable(
                hints.split(":")
                for hints in args.pysat_hints
            )
        ))

    if world == "office":
        experiment_l = "../experiments/office/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/office/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/office/tests/ground_truth.txt"
            experiment_t = "../experiments/office/tests/ground_truth.txt"
    elif world == "craft":
        experiment_l = "../experiments/craft/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/craft/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/craft/tests/ground_truth.txt"
            experiment_t = "../experiments/craft/tests/ground_truth.txt"
    elif world == "traffic":
        experiment_l = "../experiments/traffic/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/traffic/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/traffic/tests/ground_truth.txt"
            experiment_t = "../experiments/traffic/tests/ground_truth.txt"
    elif world == "taxi":
        experiment_l = "../experiments/taxi/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/taxi/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/taxi/tests/ground_truth.txt"
            experiment_t = "../experiments/taxi/tests/ground_truth.txt"
    else:
        raise NotImplementedError("world={:r}".format(world))
    world += "world"


    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment_l, "num_times: " + str(num_times), show_print)
    run_experiment(world, alg_name, experiment_t, experiment_l, num_times, show_print, show_plots, al_alg_name, sat_alg_name, pysat_hints)
