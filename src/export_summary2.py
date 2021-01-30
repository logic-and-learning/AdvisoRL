#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, argparse
import csv
import copy
from run1 import (
    get_params_traffic_world,
    get_params_office_world,
    get_params_craft_world,
    get_params_taxi_world,
)


worlds_dict = {
    "traffic": {
        "params": get_params_traffic_world('../experiments/traffic/tests/ground_truth.txt'),
        "num_tasks": 1,
    },
    "office": {
        "params": get_params_office_world('../experiments/office/tests/ground_truth.txt'),
        "num_tasks": 4,
    },
    "craft": {
        "params": get_params_craft_world('../experiments/craft/tests/ground_truth.txt'),
        "num_tasks": 4,
    },
    "taxi": {
        "params": get_params_taxi_world('../experiments/taxi/tests/ground_truth.txt'),
        "num_tasks": 1,
    },
}


algs_dict = {
    "jirprpni": {
        "label": 'JIRP+RPNI',
        "color": 'black',
        # "linestyle": ':',
    },
    "jirpsat": {
        "label": 'JIRP+SAT',
        "color": 'green',
        # "linestyle": ':',
    },
    "jirppysat": {
        "label": 'JIRP+PYSAT',
        # "color": '#00aaaa',
        "color": 'orange',
    },
    "qlearning": {
        "label": 'QAS',
        "color": 'red',
    },
    "ddqn": {
        "label": 'DDQN',
        "color": 'purple',
    },
    "hrl": {
        "label": 'HRL',
        "color": 'blue',
    },
}

def repr_hint_simple(hint):
    if hint is None: return "none"
    return "+".join(h.lower() if h else "∅" for h in hint.split(":"))
    # return f'"{hint}"'
def repr_hint_set(hint):
    if hint is None: return "∅"
    return "{" + ", ".join(h.lower() if h else "∅" for h in hint.split(":")) + "}"
repr_hint = repr_hint_set

def get_alg_dict(algorithm, exclusive=True):
    algorithm = algorithm.lower()
    alg_dict = dict(algs_dict.get(algorithm, {}))
    if "jirppysat" in algorithm:
        alg_dict = dict(algs_dict["jirppysat"], **alg_dict)

        # from automata_learning_utils.pysat import sat_algorithms
        # for sat_algo in sat_algorithms:
        #     if sat_algo in algorithm:
        #         alg_dict["label"] += '+'+sat_algo.upper()
        #         break

        # if ":" in algorithm:
        #     hints_labels = algorithm.split(":", maxsplit=1)[1]
        #     alg_dict["label"] += ' w/ HINT: '+' repr_hint(hints_labels)

        if ":" in algorithm: alg_dict["label"] = repr_hint(algorithm.split(":", maxsplit=1)[1])
        else:                alg_dict["label"] = repr_hint(None)

        if not exclusive: alg_dict["color"] = None # automatic color
    return alg_dict



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts

    y.append(sum(y[-5:]) / len(y[-5:]))
    y.append(sum(y[-5:]) / len(y[-5:]))
    y.append(sum(y[-5:]) / len(y[-5:]))
    y.append(sum(y[-5:]) / len(y[-5:]))
    y.append(sum(y[-5:]) / len(y[-5:]))

    y_smooth = np.convolve(y[0:-5], box, mode='same')
    y_smooth[-1] = y_smooth[-6]
    y_smooth[-2] = y_smooth[-6]
    y_smooth[-3] = y_smooth[-6]
    y_smooth[-4] = y_smooth[-6]
    y_smooth[-5] = y_smooth[-6]
    return y_smooth




def export_results(world, task_id, algorithm, subplots=False,
    show=True, output_file="../plotdata/figure.png",
):
    files = os.listdir("../plotdata/")

    world_dict = worlds_dict[world]

    if isinstance(task_id, str):
        task_id = int(task_id)
    if task_id == 0:
        tasks = list(range(1,world_dict["num_tasks"]+1))
    elif isinstance(task_id, int):
        tasks = [task_id]
    else:
        tasks = task_id

    if algorithm == "all":
        algorithms = list(algs_dict.keys())
    elif isinstance(algorithm, str):
        algorithms = [algorithm]
    else:
        algorithms = algorithm

    if len(algorithms) <= 1:
        subplots = False

    percentiles = (10,25,50,75,90)

    step_unit = world_dict["params"][0].num_steps
    max_step  = world_dict["params"][3].total_steps

    steps = np.linspace(0, max_step, (max_step // step_unit) + 1, endpoint=True)

    data_dict = {
        algorithm: {
            0: { # step = 0
                task: [0 for p in percentiles]
                for task in tasks
            },
        }
        for algorithm in algorithms
    }
    for task in tasks:
        files_of_interest = [
            file for file in files
            if f"{world}world{task}" in file
            if ".csv" in file
        ]
        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            for algorithm in algorithms:
                if not f"{task}{algorithm}.csv" in file:
                    continue
                if "detail" in file: continue
                with open(file_str) as csvfile:
                    step = 0
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        step += step_unit
                        p_dict = data_dict[algorithm].setdefault(step, {})
                        p_dict[task] = [np.percentile(row, p) for p in percentiles]
                break

    data_list = {
        algorithm: []
        for algorithm in data_dict.keys()
    }
    for step in steps:
        for algorithm,data_dict_algorithm in data_dict.items():
            p_dict = data_dict_algorithm.get(step, None)
            if p_dict is None: continue
            data_list[algorithm].append(
                [
                    sum(p_tasks)/len(p_tasks)
                    for p_tasks in zip(*(
                        ps for task,ps in p_dict.items()
                    ))
                ]
            )

    if not subplots:
        fig, ax = plt.subplots(1,1)
        axes = [ax]*len(data_list)
        fig.set_figheight(6)
        fig.set_figwidth(8)
    else:
        fig, axes = plt.subplots(len(data_list),1)
        fig.set_figheight(6*len(data_list))
        fig.set_figwidth(8)


    exclusive = subplots or len(algorithms) == 1
    for (algorithm, ps), ax in zip(data_list.items(), axes):
        # print(ps)
        ps = [smooth(list(p), 5) for p in zip(*ps)]

        steps = np.linspace(0, (len(ps[0]) - 1) * step_unit, len(ps[0]), endpoint=True)
        ax.set_xlim(0, (len(ps[0]) - 1) * step_unit)

        p10,p25,p50,p75,p90 = ps

        alg_dict = get_alg_dict(algorithm, exclusive)
        p, = ax.plot(steps, p50, alpha=1, **alg_dict)
        color = p.get_color()
        # ax.plot(steps, p25, color=color, alpha=0)
        # ax.plot(steps, p75, color=color, alpha=0)

        # ax.fill_between(steps, p50, p25, color=color, alpha=0.25)
        # ax.fill_between(steps, p50, p75, color=color, alpha=0.25)
        if exclusive:
            ax.fill_between(steps, p10, p90, facecolor=color, alpha=0.05)
            ax.fill_between(steps, p25, p75, facecolor=color, alpha=0.25)
        else:
            ax.fill_between(steps, p25, p75, facecolor=color, alpha=0.10)

    for ax in axes:
        ax.grid(True)

        ax.set_xlabel('number of training steps', fontsize=28)
        ax.set_ylabel('reward', fontsize=28)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, max_step)

        ax.locator_params(axis='x', nbins=5)

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        if not subplots:
            plt.gcf().subplots_adjust(bottom=0.15)
        # plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '','','D-DQN','', '', 'HRL', ''))
        # ax.legend(alg["label"] for alg in algs_dict.values())
        ax.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 18})

        ax.tick_params(axis='both', which='major', labelsize=22)

    if output_file is not None:
        output_file = output_file.format(
            world=world, task=",".join(str(task) for task in tasks),
        )
        plt.savefig(output_file, dpi=600, metadata={'CreationDate':None})
    if show: plt.show()
    else:    plt.close()

if __name__ == "__main__":


    # EXAMPLE: python3 export_summary.py --world="craft"

    # Getting params
    worlds     = list(worlds_dict.keys())
    algorithms = list(algs_dict.keys()) + ["all"]

    print("Note: ensure that runs correspond with current parameters for curriculum.total_steps and testing_params.num_steps!")
    print("")

    parser = argparse.ArgumentParser(prog="export_summary", description='After running the experiments, this algorithm computes a summary of the results.')
    parser.add_argument('--world', default='traffic', type=str,
                        help='This parameter indicated which world to solve.')
    parser.add_argument('--algorithms', action='append',
                        help='This parameter indicated which algorithm(s) to solve. Set to "all" to graph all methods. Separate multiple algorithms with a comma.')
    parser.add_argument('--tasks', action='append',
                        help='This parameter indicates which task(s) to display. Set to zero to graph all tasks. Separate multiple tasks with a comma.')
    parser.add_argument('--subplots', action="store_true",
                        help='Plot each algorithm in a distinct subplot.')

    args = parser.parse_args()
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")


    # Computing the experiment summary
    world = args.world
    args_tasks = args.tasks if args.tasks else ['1']
    tasks = [
        int(t.strip("\'\" "))
        for arg_task in args_tasks
        for t in arg_task.strip("[]() ").split(",")
    ]
    args_algorithms = args.algorithms if args.algorithms else ['jirpsat']
    algorithms = [
        t.strip("\'\" ").lower()
        for arg_algorithm in args_algorithms
        for t in arg_algorithm.strip("[]() ").split(",")
    ]
    export_results(world, tasks, algorithms, args.subplots)
