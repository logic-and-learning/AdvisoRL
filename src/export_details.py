#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import Button
import numpy as np
import sys, os, argparse
import csv
import copy
import contextlib
# from run1 import get_params_traffic_world, get_params_office_world, get_params_craft_world
from pprint import pprint

import os, io, tarfile, zipfile, functools
def close_together(obj, *others):
    old_close = obj.close
    @functools.wraps(old_close)
    def new_close(*arg, **kwargs):
        old_close(*arg, **kwargs)
        for other in others:
            other.close()
    obj.close = new_close
    return obj
def deep_open(file):
    """Open files inside archives too!"""
    sub = None
    while not os.path.isfile(file):
        file, subroot = os.path.split(file)
        if sub is None: sub = subroot
        else: sub = os.path.join(subroot, sub)
    if sub is None:
        return open(file)
    if tarfile.is_tarfile(file):
        tar = tarfile.open(file)
        f = io.TextIOWrapper(tar.extractfile(sub))
        return close_together(f, tar)
    if zipfile.is_zipfile(file):
        zip = zipfile.ZipFile(file)
        f = io.TextIOWrapper(zip.open(file))
        return close_together(f, zip)

types_parameters = {
    "world":         str,
    "task":          int,
    "algorithm":     str,
    "enter_loop":    int,
    "step_unit":     int,
    "total_steps":   int,
    "epsilon":       float,
    "al_algorithm":  str,
    "sat_algorithm": str,
    "hint_at":       str,
    "hint":          str,
}
types_details = {
    "task_time":   float,
    "al_step":     [int],
    "al_pos":      [int],
    "al_neg":      [int],
    "al_time":     [float],
    "total_time":  float,
    "reward_step": int,
}
def read_file(detail_file):
    with deep_open(detail_file) as csvfile:
        readcsv = csv.reader(csvfile)
        parameters = {}
        details = {}
        mode, types = "", {}
        for row in readcsv:
            if not len(row): continue
            head, row = row[0], row[1:]
            key = head.strip().strip(":")
            if "RUN_PARAMETERS" in head:
                mode, types = 'PARAMETERS', types_parameters
                continue
            if "ITERATION_DETAIL" in head:
                mode, types = 'DETAIL', types_details
                detail = details.setdefault(int(row[0]), {})
                continue
            typ = types.get(key, [str])
            if isinstance(typ, list): # convert to list
                row = list(map(typ[0], row))
                row = np.array(row)
            else: # convert to single
                try:
                    row = typ(row[0])
                except Exception:
                    row = None
            if mode == 'PARAMETERS':
                parameters[key] = row
            elif mode == 'DETAIL':
                detail[key] = row
    return parameters, details

def repr_hint_simple(hint):
    if hint is None: return "none"
    return "&".join(h.lower() if h else "∅" for h in hint.split(":"))
    # return f'"{hint}"'
def repr_hint_set(hint):
    if hint is None: return "∅"
    return "{" + ", ".join(h.lower() if h else "∅" for h in hint.split(":")) + "}"
repr_hint = repr_hint_set

def plot_run(ax, *data, color=None, label=None):
    p, = ax.plot(*data,
        "-", color=color, alpha=0.3,
        zorder=2,
    )
    color = p.get_color()
    ax.plot(*(d[:-1] for d in data), # all but last one
        "o", color=color, markeredgecolor="black",
        markeredgewidth=0.2, markersize=5,
        zorder=3,
        # label=label,
    )
    ax.plot(*([d[-1],d[-1]] for d in data), # last one
        "o", color=color, markeredgecolor="black",
        markeredgewidth=1, markersize=7,
        zorder=4,
        label=label,
    )
    return color


def plot_step_time(ax, details, *, monochrome=False, label=None):
    ax.set_xlabel('training step')
    # ax.set_ylabel('AL computation time (s)')
    # ax.set_ylabel('automaton learning computation time (s)')
    ax.set_ylabel('computation time (s)')
    ax.grid(True, which="major")


    # ax.set_xscale('log')
    # ax.grid(True, axis="x", which="minor", alpha=0.2)

    ax.set_yscale('log')
    ax.grid(True, axis="y", which="minor", alpha=0.2)

    ax.set_ylim(2e-3, 2e4) # time

    color = None
    for t,detail in details.items():
        steps = detail["al_step"]
        times = detail["al_time"]
        if not monochrome: color = None
        color = plot_run(ax, steps, times, color=color, label=label)
        label = None # only the first one

def plot_sample_time(ax, details, *, monochrome=False, label=None):
    ax.set_xlabel('sample size')
    # ax.set_ylabel('AL computation time (s)')
    # ax.set_ylabel('automaton learning computation time (s)')
    ax.set_ylabel('computation time (s)')
    ax.grid(True, which="major")

    ax.set_xscale('log')
    ax.grid(True, axis="x", which="minor", alpha=0.2)

    ax.set_yscale('log')
    ax.grid(True, axis="y", which="minor", alpha=0.2)

    ax.set_xlim(5e0, 2e4) # sample
    ax.set_ylim(2e-3, 2e4) # time

    color = None
    for t,detail in details.items():
        sample_sizes = detail["al_pos"] + detail["al_neg"]
        times = detail["al_time"]
        if not monochrome: color = None
        color = plot_run(ax, sample_sizes, times, color=color, label=label)
        label = None # only the first one

def plot_step_sample(ax, details, *, monochrome=False, label=None):
    ax.set_xlabel('training step')
    ax.set_ylabel('sample size')
    ax.grid(True, which="major")

    # ax.set_xscale('log')
    # ax.grid(True, axis="x", which="minor", alpha=0.2)

    # ax.set_yscale('log')
    # ax.grid(True, axis="y", which="minor", alpha=0.2)

    color = None
    for t,detail in details.items():
        steps = detail["al_step"]
        sample_sizes = detail["al_pos"] + detail["al_neg"]
        if not monochrome: color = None
        color = plot_run(ax, steps, sample_sizes, color=color, label=label)
        label = None # only the first one


def plot_step_sample_time(ax3d, details, *, monochrome=False, label=None):
    ax3d.set_xlabel('training step')
    ax3d.set_ylabel('sample size')
    # ax3d.set_zlabel('AL time (s)')
    ax3d.set_zlabel('computation time (s)')
    isxlog = isylog = iszlog = False

    isylog = True
    # ax3d.set_ylabel(ax3d.get_ylabel()+' (log)')
    ax3d.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3d.yaxis.set_major_formatter(ticker.FormatStrFormatter("$10^{%d}$"))

    iszlog = True
    # ax3d.set_zlabel(ax3d.get_zlabel()+' (log s)')
    ax3d.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3d.zaxis.set_major_formatter(ticker.FormatStrFormatter("$10^{%d}$"))

    # does not work
    # ax3d.set_xscale('log')
    # ax3d.set_yscale('log')
    # ax3d.set_zscale('log')

    ax3d.set_proj_type('ortho')
    # ax3d.view_init(  0,-90) # z(x)
    # ax3d.view_init(  0,  0) # z(y)
    # ax3d.view_init( 90,-90) # y(x)

    color = None
    for t,detail in details.items():
        steps = detail["al_step"]
        sample_sizes = detail["al_pos"] + detail["al_neg"]
        times = detail["al_time"]

        if isxlog: steps = np.log10(steps) # log scale
        if isylog: sample_sizes = np.log10(sample_sizes) # log scale
        if iszlog: times = np.log10(times) # log scale

        if not monochrome: color = None
        color = plot_run(ax3d, steps, sample_sizes, times, color=color, label=label)
        label = None # only the first one



def export_learning_results(world, task, algorithm, satalgorithm, hints, view, *,
    detail_file="../plotdata/details_{run_name}.csv",
    show=True, output_file="../plotdata/details-{view}-{satalgorithm}.png",
):
    """Plot each relearn for each run."""
    algo_name = f"{algorithm}{satalgorithm}"

    # fig, axes = plt.subplots(2,1)
    fig, axes = plt.subplots(1,1)
    axes = np.array(axes).flatten()
    fig.canvas.set_window_title(f"{world}world{task}{algo_name}")
    fig.patch.set_alpha(0)
    # for ax in axes: ax.set_facecolor('xkcd:salmon')
    fig.set_figheight(5)
    fig.set_figwidth(6)

    if view == "3D":
        from mpl_toolkits.mplot3d import Axes3D
        axes[0].remove()
        axes[0] = ax3d = fig.add_subplot(111, projection='3d')
        # axes[1].remove()
        # ax3d = fig.add_subplot(212, projection='3d')
        axxz = plt.axes([0.10, 0.025, 0.19, 0.05])
        axyz = plt.axes([0.30, 0.025, 0.19, 0.05])
        axxy = plt.axes([0.50, 0.025, 0.19, 0.05])
        bxz = Button(axxz, 'time(step)')
        byz = Button(axyz, 'time(sample)')
        bxy = Button(axxy, 'sample(step)')
        bxz.on_clicked(lambda e: ax3d.view_init(  0,-90))
        byz.on_clicked(lambda e: ax3d.view_init(  0,  0))
        bxy.on_clicked(lambda e: ax3d.view_init( 90,-90))

    def relstd(array):
        return np.std(array) / np.abs(np.mean(array))
    def stdpercent(array):
        return "{:.1f}%".format(relstd(array)*100)

    print(f"{world} {task} {algo_name}")
    for hint in hints:
        algo = f"{algo_name}:{hint}" if hint is not None else f"{algo_name}"
        run_name = f"{world}world{task}{algo}"
        detail_file = f"../plotdata/details_{run_name}.csv"
        parameters, details = read_file(detail_file.format(run_name=run_name))

        print(f">>> {repr_hint(hint)}")
        max_step = parameters['total_steps']

        if view == "step-vs-time": plot_step_time(axes[0], details, label=repr_hint(hint), monochrome=1)
        if view == "sample-vs-time": plot_sample_time(axes[0], details, label=repr_hint(hint), monochrome=1)
        if view == "step-vs-sample": plot_step_sample(axes[0], details, label=repr_hint(hint), monochrome=1)

        if view == "3D": plot_step_sample_time(ax3d, details, label=repr_hint(hint), monochrome=1)

        if view in ["step-vs-time" ,"step-vs-sample" ,"3D"]:
            axes[0].set_xlim(0, max_step)
        # plot_step_sample_time(ax3d, details, monochrome=0)

    axes[0].legend()

    if output_file is not None:
        output_file = output_file.format(
            world=world, task=task, #hints=",".join(hints),
            algorithm=algorithm, satalgorithm=satalgorithm,
            view=view,
        )
        plt.savefig(output_file, dpi=600, metadata={'CreationDate':None})
    if show: plt.show()
    else:    plt.close()
export_learning_results.views = [
    "step-vs-time",
    "sample-vs-time",
    "step-vs-sample",
    # "3D",
]

def export_median_results(world, task, algorithm, satalgorithm, hints, view, *,
    detail_file="../plotdata/details_{run_name}.csv",
    show=True, output_file="../plotdata/median-{view}-{satalgorithm}.png"
):
    """Boxplot views."""
    algo_name = f"{algorithm}{satalgorithm}"

    # fig, axes = plt.subplots(2,1)
    fig, axes = plt.subplots(1,1)
    axes = np.array(axes).flatten()
    fig.canvas.set_window_title(f"{world}world{task}{algo_name}")
    fig.patch.set_alpha(0)
    fig.set_figheight(5)
    fig.set_figwidth(6)

    def alternative(*values, avoid=None):
        """get first not None value"""
        return next(value for value in values if value is not avoid)

    print(f"{world} {task} {algo_name}")

    d_task_times = {}
    d_als_times = {}
    d_alN_times = {}
    d_reward_steps = {}

    last_als_medtime = 0
    for hint in hints:
        algo = f"{algo_name}:{hint}" if hint is not None else f"{algo_name}"
        run_name = f"{world}world{task}{algo}"
        parameters, details = read_file(detail_file.format(run_name=run_name))

        print(f">>> {repr_hint(hint)}")
        max_step = parameters['total_steps']
        task_times = np.array([detail['task_time'] for detail in details.values()], dtype=float)
        als_times  = np.array([sum(detail['al_time']) for detail in details.values()])
        alN_times  = np.array([detail['al_time'][-1] for detail in details.values()])
        tot_times  = task_times + als_times
        al_relearns   = np.array([len(detail['al_time']) for detail in details.values()])
        sample_last = np.array([detail['al_pos'][-1]+detail['al_neg'][-1] for detail in details.values()])
        sample_0    = np.array([detail['al_pos'][0]+detail['al_neg'][0] for detail in details.values()])
        sample_1    = np.array([detail['al_pos'][1]+detail['al_neg'][1] for detail in details.values() if len(detail['al_pos'])>1])
        als_medtime = np.median(als_times)
        reward_steps = np.array([alternative(detail['reward_step'], max_step) for detail in details.values()])# WARNING: we assume convergence in max_step which is wrong
        is_reward_steps_approx = any(detail['reward_step'] is None for detail in details.values())

        h = repr_hint(hint)
        d_task_times[h] = task_times
        d_als_times[h] = als_times
        d_alN_times[h] = alN_times
        d_reward_steps[h] = reward_steps

    d = {
        "als-time":  d_als_times,
        "alN-time":  d_alN_times,
        "task-time": d_task_times,
        "reward-step": d_reward_steps,
    }[view]

    ax = axes[0]

    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.gcf().subplots_adjust(left=0.20, bottom=0.25, right=0.90, top=0.95)

    if "-time" in view:
        ax.boxplot(list(d.values()), labels=d.keys(),
            whis=[5, 95],
            # whis="range",
            vert=True,
        )
        ax.set_ylabel('computation time (s)', fontsize=22)
        ax.grid(True, axis="y", which="major")

        ax.set_yscale('log')
        ax.grid(True, axis="y", which="minor", alpha=0.2)
        ax.set_ylim(1e-2,1e4)

        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=9))
        ax.tick_params(axis='x', which='major', labelrotation=90)

    elif "-step" in view:
        ax.boxplot(list(d.values()), labels=d.keys(),
            whis=[5, 95],
            # whis="range",
            vert=False,
        )
        ax.set_xlabel('number of training steps', fontsize=22)
        ax.grid(True, axis="x", which="major")
        ax.set_xlim(0,parameters['total_steps'])
        ax.locator_params(axis='x', nbins=5)

        ax.tick_params(axis="x", which='major')

    # ax.tick_params(axis='both', which='major', labelsize=22)

    # ax.set_ylim(0,110)


    # for h,x in d_alN_times.items():
    #     ax.hist(x, 5000, density=True, histtype='step', orientation='horizontal',
    #         cumulative=True, label=h,
    #     )
    # ax.set_ylabel('computation time (s)')
    # # ax.grid(True, axis="y", which="major")
    # ax.set_yscale('log')
    # # ax.grid(True, axis="y", which="minor", alpha=0.2)
    # ax.set_ylim(1e-2,1e4)
    # ax.legend()

    if output_file is not None:
        output_file = output_file.format(
            world=world, task=task, #hints=",".join(hints),
            algorithm=algorithm, satalgorithm=satalgorithm,
            view=view,
        )
        plt.savefig(output_file, dpi=600, metadata={'CreationDate':None})
    if show: plt.show()
    else:    plt.close()
export_median_results.views = [
    "als-time", # cumulative automaton relearn computation time
    "alN-time", # last automaton relearn computation time
    "task-time", # episode computation time
    "reward-step", # optimal policies times
]

def print_median_results(world, task, algorithm, satalgorithm, hints, *,
    detail_file="../plotdata/details_{run_name}.csv",
    file=sys.stdout,
):
    with contextlib.redirect_stdout(file):
        algo_name = f"{algorithm}{satalgorithm}"

        def _relstd(array):
            """standard deviation relative to the average"""
            return np.std(array) / np.abs(np.mean(array))
        # def _relmed(array):
        #     """Median relative to no hint"""
        #     return np.median(array) / np.median(...)

        def relpercent(value):
            return "{:5.1f}%".format(value*100)
        def relstd(array):
            return relpercent(_relstd(array))+"avg"
        # def relmed(array):
        #     return relpercent(_relmed(array))+repr_hint(None)
        def alternative(*values, avoid=None):
            """get first not None value"""
            return next(value for value in values if value is not avoid)

        print(f"========== {world} {task} {algo_name} ==========")
        # last_als_medtime = 0
        for hint in hints:
            algo = f"{algo_name}:{hint}" if hint is not None else f"{algo_name}"
            run_name = f"{world}world{task}{algo}"
            parameters, details = read_file(detail_file.format(run_name=run_name))

            print(f">>> {repr_hint(hint)}")
            max_step = parameters['total_steps']
            task_times = np.array([detail['task_time'] for detail in details.values()], dtype=float)
            als_times  = np.array([sum(detail['al_time']) for detail in details.values()])
            alN_times  = np.array([detail['al_time'][-1] for detail in details.values()])
            tot_times  = task_times + als_times
            al_relearns   = np.array([len(detail['al_time']) for detail in details.values()])
            sample_last = np.array([detail['al_pos'][-1]+detail['al_neg'][-1] for detail in details.values()])
            sample_0    = np.array([detail['al_pos'][0]+detail['al_neg'][0] for detail in details.values()])
            sample_1    = np.array([detail['al_pos'][1]+detail['al_neg'][1] for detail in details.values() if len(detail['al_pos'])>1])
            als_medtime = np.median(als_times)
            reward_steps = np.array([alternative(detail['reward_step'], max_step) for detail in details.values()])# WARNING: we assume convergence in max_step which is wrong
            is_reward_steps_approx = any(detail['reward_step'] is None for detail in details.values())
            # print ("...prev ALs: medratio = {:7.3f}".format(last_als_medtime / als_medtime))
            # last_als_medtime = als_medtime
            print ("number of independant runs: {}".format(len(tot_times)))
            props = [np.median, np.mean, np.std, relstd]
            print ("time:   med = {:7.3f}s   avg = {:7.3f}s   std = {:4f}({})".format(*[f(tot_times) for f in props]))
            print ("- task: med = {:7.3f}s   avg = {:7.3f}s   std = {:4f}({})".format(*[f(task_times) for f in props]))
            print ("- ALs:  med = {:7.3f}s   avg = {:7.3f}s   std = {:4f}({})".format(*[f(als_times) for f in props]))
            print ("number of relearns: med={:3.1f}  avg={:5.3f}  std={:4f}({})".format(*[f(al_relearns) for f in props]))
            print ("convergence step: med{e}{:.0f}  avg{e}{:.0f}  std{e}{:4f}({})".format(*[f(reward_steps) for f in props], e="=~"[is_reward_steps_approx]))
            # print ("sample size:  med = {:6.1f}   avg = {:6.1f}   std = {:4f}({})".format(*[f(sample_last) for f in [np.median, np.mean, np.std, relstd]]))
            # # print ("sample0 size: med = {:6.1f}   avg = {:6.1f}   std = {:4f}({})".format(*[f(sample_0)    for f in [np.median, np.mean, np.std, relstd]]))
            # print ("sample1 size: med = {:6.1f}   avg = {:6.1f}   std = {:4f}({})".format(*[f(sample_1)    for f in [np.median, np.mean, np.std, relstd]]))



if __name__ == '__main__':

    world="office"
    task=7
    hints = [
        None,
        # "",
        "b",
        # "d",
        # "g",
        # "b:d:g",
        "bd",
        "bdb",
        "bdbg",
        "a:b",
    ]

    algorithm="jirppysat"
    satalgorithm="rc2"
    # satalgorithm="fm"

    # for view in [1,2,3]:
    #     for satalgorithm in ["fm","rc2"]:

    # export_median_results(
    #     world, task,
    #     algorithm, satalgorithm,
    #     view="als-time",
    #     hints=hints,
    # )

    # export_learning_results(world, task, algorithm, satalgorithm, hints=hints, view="3D")
    export_median_results(world, task, algorithm, satalgorithm, hints=hints, view="als-time")
    export_median_results(world, task, algorithm, satalgorithm, hints=hints, view="reward-step")

    # for satalgorithm in [
    #     # "fm",
    #     "rc2",
    # ]:
    #     for view in export_learning_results.views:
    #         export_learning_results(
    #             world, task,
    #             algorithm, satalgorithm,
    #             hints=hints,
    #             view=view,
    #             show=False,
    #         )
    #     for view in export_median_results.views:
    #         export_median_results(
    #             world, task,
    #             algorithm, satalgorithm,
    #             hints=hints,
    #             view=view,
    #             show=False,
    #         )
    #         # break


    pass
