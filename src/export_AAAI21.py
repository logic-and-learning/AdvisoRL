#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os

import termcolor
def warning(text): return termcolor.colored(text, color='red', attrs=["bold"])

from export_summary2 import *
from export_details import *

data_location="../plotdata"
latex_location=".."

def export_case_1_perfs(ext="png"):
    world, task = "office", 7
    algorithm, satalgorithm = "jirppysat", "rc2"
    hints = [
        None,
        # "",
        "b",
        "d",
        "g",
        "b:d:g",
        "bd",
        "bdb",
        "bdbg",
    ]
    other_algorithms = [
        "jirpsat",
        "hrl",
        "ddqn",
    ]
    plotdata_location = os.path.join(data_location, "plotdata_200606_2.7-details.tar.gz/plotdata")
    export_location = os.path.join(latex_location, "figures", "office-t7")
    # TODO: extract archive into plotdata
    print(f"MAKE SURE '{warning(plotdata_location)}' ARE EXTRACTED INSIDE '../plotdata'")

    print()
    algorithms = [algorithm+satalgorithm+("" if hint is None else ":"+hint) for hint in hints]
    output_file = f"perf-{algorithm}.{ext}"
    print(f"# EQUIVALENT EXPORT COMMAND:")
    print(f"# {output_file}")
    print(f"python3 export_summary2.py --world={world} --task={task}" + ''.join(" --algo="+algo for algo in algorithms))
    export_results(world, task, algorithms,
        subplots=False,
        show=False, output_file=os.path.join(export_location, output_file),
    )

    for algorithm in other_algorithms:
        output_file = f"perf-{algorithm}.{ext}"
        print(f"# {output_file}")
        print(f"python3 export_summary2.py --world={world} --task={task} --algo={algorithm}")
        export_results(world, task, algorithm,
            subplots=False,
            show=False, output_file=os.path.join(export_location, output_file),
        )
    print()

def export_aux_case_1_perfs(ext="png"):
    # raise RuntimeError("Don't use this function, plotdata_location is wrong") # we might want to re-run it
    world, task = "office", 7
    algorithm = "jirppysat"
    satalgorithms = [
        # "gc3", # TODO: compute in --num=30
        "fm",
        "rc2",
    ]
    hints = [
        None,
        # "",
        "b",
        "bd",
        "bdb",
        "bdbg",
    ]
    plotdata_location = os.path.join(data_location, "plotdata_200606_2.7-details.tar.gz/plotdata")
    export_location = os.path.join(latex_location, "figures", "office-t7")
    # TODO: extract archive into plotdata
    print(f"MAKE SURE '{warning(plotdata_location)}' ARE EXTRACTED INSIDE '../plotdata'")

    for satalgorithm in satalgorithms:
        algorithms = [algorithm+satalgorithm+("" if hint is None else ":"+hint) for hint in hints]
        output_file = f"perf-{algorithm}-{satalgorithm}.{ext}"
        print(f"# {output_file}")
        print(f"python3 export_summary2.py --world={world} --task={task}" + ''.join(" --algo="+algo for algo in algorithms))
        export_results(world, task, algorithms,
            subplots=False,
            show=False, output_file=os.path.join(export_location, output_file),
        )
    print()

def export_case_1_details(ext="png"):
    world, task = "office", 7
    hints = [
        None,
        # "",
        "b",
        "d",
        "g",
        "b:d:g",
        "bd",
        "bdb",
        "bdbg",
    ]
    algorithm, satalgorithm = "jirppysat", "rc2"
    plotdata_location = os.path.join(data_location, "plotdata_200606_2.7-details.tar.gz/plotdata")
    detail_file = os.path.join(plotdata_location, "details_{run_name}.csv")
    export_location = os.path.join(latex_location, "figures", "office-t7")

    # for view in export_learning_results.views:
    #     export_learning_results(
    #         world, task,
    #         algorithm, satalgorithm, hints=hints,
    #         view=view,
    #         detail_file=detail_file),
    #         show=False, output_file=os.path.join(export_location, "details-{view}."+ext),
    #     )
    for view in export_median_results.views:
        export_median_results(
            world, task,
            algorithm, satalgorithm, hints=hints,
            view=view,
            detail_file=detail_file,
            show=False, output_file=os.path.join(export_location, "median-{view}."+ext),
        )
    print_median_results(
        world, task,
        algorithm, satalgorithm, hints=hints,
        detail_file=detail_file,
    )

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
    ]

    # for satalgorithm in [
    #     "fm",
    #     "rc2",
    # ]:
    #     for view in export_median_results.views:
    #         export_median_results(
    #             world, task,
    #             algorithm, satalgorithm, hints=hints,
    #             view=view,
    #             detail_file=os.path.join(plotdata_location, "details_{run_name}.csv"),
    #             show=False, output_file=os.path.join(export_location, "median-{view}-{satalgorithm}."+ext),
    #         )

def export_aux_case_1_details(ext="png"):
    world, task = "office", 7
    hints = [
        None,
        # "",
        "a:b", # incompatible: a
        # "b:c", # incompatible: c
        "b:e", # incompatible: e
        # "b:f", # incompatible: f
        "b",
        # "d",
        # "g",
        # "b:d:g",
        "bc:bd", # incompatible: bc
        "ba:bd", # incompatible: ba
        "bd:cd", # incompatible: cd
        "ad:bd", # incompatible: ad
        "bd",
        "bdb",
        "bdbg",
    ]
    algorithm, satalgorithm = "jirppysat", "rc2"
    plotdata_location = os.path.join(data_location, "plotdata_200606_2.7-details.tar.gz/plotdata")
    detail_file = os.path.join(plotdata_location, "details_{run_name}.csv")
    export_location = os.path.join(latex_location, "figures", "office-t7")

    # for view in export_learning_results.views:
    #     export_learning_results(
    #         world, task,
    #         algorithm, satalgorithm, hints=hints,
    #         view=view,
    #         detail_file=detail_file),
    #         show=False, output_file=os.path.join(export_location, "details-{view}-incompatible."+ext),
    #     )
    for view in ["als-time", "reward-step"]:
        export_median_results(
            world, task,
            algorithm, satalgorithm, hints=hints,
            view=view,
            detail_file=detail_file,
            show=False, output_file=os.path.join(export_location, "median-{view}-incompatible."+ext),
        )
    print_median_results(
        world, task,
        algorithm, satalgorithm, hints=hints,
        detail_file=detail_file,
    )

def export_case_2_perfs(ext="png"):
    world, task = "taxi", 1
    algorithm, satalgorithm = "jirppysat", "rc2"
    hints = [
        None,
        "e",
        "a",
        "b",
        "f",
        "eabf",
    ]
    other_algorithms = [
        "jirpsat",
        "hrl",
        "ddqn",
    ]
    plotdata_location = os.path.join(data_location, "plotdata_200700_4.1.tar.gz/plotdata")
    detail_file = os.path.join(plotdata_location, "details_{run_name}.csv")
    export_location = os.path.join(latex_location, "figures", "taxi")
    # TODO: extract archive into plotdata
    print(f"MAKE SURE '{warning(plotdata_location)}' ARE EXTRACTED INSIDE '../plotdata'")

    print()
    algorithms = [algorithm+satalgorithm+("" if hint is None else ":"+hint) for hint in hints]
    output_file = f"perf-t{task}-{algorithm}.{ext}"
    print(f"# EQUIVALENT EXPORT COMMAND:")
    print(f"# {output_file}")
    print(f"python3 export_summary2.py --world={world} --task={task}" + ''.join(" --algo="+algo for algo in algorithms))
    export_results(world, task, algorithms,
        subplots=False,
        show=False, output_file=os.path.join(export_location, output_file),
    )

    for algo in other_algorithms:
        output_file = f"perf-t{task}-{algo}.{ext}"
        print(f"# {output_file}")
        print(f"python3 export_summary2.py --world={world} --task={task} --algo={algo}")
        export_results(world, task, algo,
            subplots=False,
            show=False, output_file=os.path.join(export_location, output_file),
        )
    print()

    print_median_results(
        world, task,
        algorithm, satalgorithm, hints=hints,
        detail_file=detail_file,
    )

if __name__ == '__main__':
    for ext in [
        # "png",
        "pdf",
    ]:
        pass

        # export_case_1_perfs(ext=ext)
        # export_case_1_details(ext=ext)
        # export_case_2_perfs(ext=ext)

        # export_aux_case_1_perfs(ext=ext)
        export_aux_case_1_details(ext=ext)
