import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os, argparse
import csv
from run1 import get_params_office_world, get_params_traffic_world, get_params_craft_world

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts

    y.append(sum(y[-5:])/len(y[-5:]))
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

def export_results_traffic_world(task_id, algorithm):
    files = os.listdir("../plotdata/")

    step_unit = get_params_traffic_world('../experiments/traffic/tests/ground_truth.txt')[0].num_steps
    max_step = get_params_traffic_world('../experiments/traffic/tests/ground_truth.txt')[3].total_steps

    steps = np.linspace(0, max_step, (max_step / step_unit) + 1, endpoint=True)

    if task_id>0:
        p25 = [0]
        p50 = [0]
        p75 = [0]
        p25s = [0]
        p50s = [0]
        p75s = [0]
        p25_q = [0]
        p50_q = [0]
        p75_q = [0]
        p25_hrl = [0]
        p50_hrl = [0]
        p75_hrl = [0]
        p25_dqn = [0]
        p50_dqn = [0]
        p75_dqn = [0]
        files_of_interest = list()
        for file in files:
            if (("traffic" in file) and (".csv" in file) and (str(task_id) in file)):
                files_of_interest.append(file)

        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            if 'qlearning' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_q.append(np.percentile(row, 25))
                        p50_q.append(np.percentile(row, 50))
                        p75_q.append(np.percentile(row, 75))
            elif 'hrl' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_hrl.append(np.percentile(row, 25))
                        p50_hrl.append(np.percentile(row, 50))
                        p75_hrl.append(np.percentile(row, 75))
            elif 'dqn' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_dqn.append(np.percentile(row, 25))
                        p50_dqn.append(np.percentile(row, 50))
                        p75_dqn.append(np.percentile(row, 75))
            elif 'rpni' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25.append(np.percentile(row, 25))
                        p50.append(np.percentile(row, 50))
                        p75.append(np.percentile(row, 75))
            elif 'sat' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25s.append(np.percentile(row, 25))
                        p50s.append(np.percentile(row, 50))
                        p75s.append(np.percentile(row, 75))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)


        if algorithm == "jirprpni" or algorithm == "all":
            p25 = smooth(p25, 5)
            p50 = smooth(p50, 5)
            p75 = smooth(p75, 5)

            steps = np.linspace(0, (len(p25)-1) * step_unit, len(p25), endpoint=True)
            plt.xlim(0, (len(p25)-1) * step_unit)

            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP RPNI')
            ax.plot(steps, p75, alpha=0)

            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "jirpsat" or algorithm == "all":
            p25s = smooth(p25s, 5)
            p50s = smooth(p50s, 5)
            p75s = smooth(p75s, 5)

            steps = np.linspace(0, (len(p25s)-1) * step_unit, len(p25s), endpoint=True)
            plt.xlim(0, (len(p25s) - 1) * step_unit)

            ax.plot(steps, p25s, alpha=0)
            ax.plot(steps, p50s, color='green', label='JIRP SAT')
            ax.plot(steps, p75s, alpha=0)

            plt.fill_between(steps, p50s, p25s, color='green', alpha=0.25)
            plt.fill_between(steps, p50s, p75s, color='green', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            p25_q = smooth(p25_q, 5)
            p50_q = smooth(p50_q, 5)
            p75_q = smooth(p75_q, 5)

            steps = np.linspace(0, (len(p25_q)-1) * step_unit, len(p25_q), endpoint=True)
            plt.xlim(0, (len(p25_q) - 1) * step_unit)

            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)

            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            p25_hrl = smooth(p25_hrl, 5)
            p50_hrl = smooth(p50_hrl, 5)
            p75_hrl = smooth(p75_hrl, 5)

            steps = np.linspace(0, (len(p25_hrl)-1) * step_unit, len(p25_hrl), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        if algorithm == "ddqn" or algorithm == "all":
            p25_dqn = smooth(p25_dqn, 5)
            p50_dqn = smooth(p50_dqn, 5)
            p75_dqn = smooth(p75_dqn, 5)

            steps = np.linspace(0, (len(p25_dqn)-1) * step_unit, len(p25_dqn), endpoint=True)
            plt.xlim(0, (len(p25_dqn)-1) * step_unit)

            ax.plot(steps, p25_dqn, alpha=0)
            ax.plot(steps, p50_dqn, color='purple', label='D-DQN')
            ax.plot(steps, p75_dqn, alpha=0)

            plt.fill_between(steps, p50_dqn, p25_dqn, color='purple', alpha=0.25)
            plt.fill_between(steps, p50_dqn, p75_dqn, color='purple', alpha=0.25)
        ax.grid()

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)

        if algorithm == "all":
            plt.xlim(0,max_step)


        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '', '', 'D-DQN','','','HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

    else:

        step = 0

        p25dict = dict()
        p50dict = dict()
        p75dict = dict()
        p25sdict = dict()
        p50sdict = dict()
        p75sdict = dict()
        p25_qdict = dict()
        p50_qdict = dict()
        p75_qdict = dict()
        p25_hrldict = dict()
        p50_hrldict = dict()
        p75_hrldict = dict()
        p25_dqndict = dict()
        p50_dqndict = dict()
        p75_dqndict = dict()

        p25 = list()
        p50 = list()
        p75 = list()
        p25s = list()
        p50s = list()
        p75s = list()
        p25_q = list()
        p50_q = list()
        p75_q = list()
        p25_hrl = list()
        p50_hrl = list()
        p75_hrl = list()
        p25_dqn = list()
        p50_dqn = list()
        p75_dqn = list()

        p25dict[0] = [0,0,0,0]
        p50dict[0] = [0,0,0,0]
        p75dict[0] = [0,0,0,0]
        p25sdict[0] = [0,0,0,0]
        p50sdict[0] = [0,0,0,0]
        p75sdict[0] = [0,0,0,0]
        p25_qdict[0] = [0,0,0,0]
        p50_qdict[0] = [0,0,0,0]
        p75_qdict[0] = [0,0,0,0]
        p25_hrldict[0] = [0,0,0,0]
        p50_hrldict[0] = [0,0,0,0]
        p75_hrldict[0] = [0,0,0,0]
        p25_dqndict[0] = [0,0,0,0]
        p50_dqndict[0] = [0,0,0,0]
        p75_dqndict[0] = [0,0,0,0]


        files_dict = dict()
        for file in files:
            if (("traffic" in file) and (".csv" in file)):
                if "1" in file:
                    task = 1
                if "2" in file:
                    task = 2
                if "3" in file:
                    task = 3
                if "4" in file:
                    task = 4

                if task not in files_dict:
                    files_dict[task] = [file]
                else:
                    files_dict[task].append(file)

            for task in files_dict:
                for file in files_dict[task]:
                    file_str = ("../plotdata/") + file
                    if 'qlearning' in file:
                        with open(file_str) as csvfile:
                            step = 0
                            readcsv = csv.reader(csvfile)

                            for row_ in readcsv:
                                if len(row_) > 1:
                                    row = list(map(int, row_))
                                else:
                                    row = [float(row_[0])]
                                step += step_unit
                                if step in p25_qdict:
                                    p25_qdict[step].append(np.percentile(row, 25))
                                    p50_qdict[step].append(np.percentile(row, 50))
                                    p75_qdict[step].append(np.percentile(row, 75))
                                else:
                                    p25_qdict[step] = [np.percentile(row, 25)]
                                    p50_qdict[step] = [np.percentile(row, 50)]
                                    p75_qdict[step] = [np.percentile(row, 75)]

                    elif 'hrl' in file:
                        with open(file_str) as csvfile:
                            step = 0
                            readcsv = csv.reader(csvfile)
                            for row_ in readcsv:
                                if len(row_) > 1:
                                    row = list(map(int, row_))
                                else:
                                    row = [float(row_[0])]
                                step += step_unit
                                if step in p25_hrldict:
                                    p25_hrldict[step].append(np.percentile(row, 25))
                                    p50_hrldict[step].append(np.percentile(row, 50))
                                    p75_hrldict[step].append(np.percentile(row, 75))
                                else:
                                    p25_hrldict[step] = [np.percentile(row, 25)]
                                    p50_hrldict[step] = [np.percentile(row, 50)]
                                    p75_hrldict[step] = [np.percentile(row, 75)]

                    elif 'dqn' in file:
                        with open(file_str) as csvfile:
                            step = 0
                            readcsv = csv.reader(csvfile)
                            for row_ in readcsv:
                                if len(row_) > 1:
                                    row = list(map(int, row_))
                                else:
                                    row = [float(row_[0])]
                                step += step_unit
                                if step in p25_dqndict:
                                    p25_dqndict[step].append(np.percentile(row, 25))
                                    p50_dqndict[step].append(np.percentile(row, 50))
                                    p75_dqndict[step].append(np.percentile(row, 75))
                                else:
                                    p25_dqndict[step] = [np.percentile(row, 25)]
                                    p50_dqndict[step] = [np.percentile(row, 50)]
                                    p75_dqndict[step] = [np.percentile(row, 75)]
                    elif 'rpni' in file:
                        with open(file_str) as csvfile:
                            step = 0
                            readcsv = csv.reader(csvfile)
                            for row_ in readcsv:
                                if len(row_) > 1:
                                    row = list(map(int, row_))
                                else:
                                    row = [float(row_[0])]
                                step += step_unit
                                if step in p25dict:
                                    p25dict[step].append(np.percentile(row, 25))
                                    p50dict[step].append(np.percentile(row, 50))
                                    p75dict[step].append(np.percentile(row, 75))
                                else:
                                    p25dict[step] = [np.percentile(row, 25)]
                                    p50dict[step] = [np.percentile(row, 50)]
                                    p75dict[step] = [np.percentile(row, 75)]
                    elif 'sat' in file:
                        with open(file_str) as csvfile:
                            step = 0
                            readcsv = csv.reader(csvfile)
                            for row_ in readcsv:
                                if len(row_) > 1:
                                    row = list(map(int, row_))
                                else:
                                    row = [float(row_[0])]
                                step += step_unit
                                if step in p25sdict:
                                    p25sdict[step].append(np.percentile(row, 25))
                                    p50sdict[step].append(np.percentile(row, 50))
                                    p75sdict[step].append(np.percentile(row, 75))
                                else:
                                    p25sdict[step] = [np.percentile(row, 25)]
                                    p50sdict[step] = [np.percentile(row, 50)]
                                    p75sdict[step] = [np.percentile(row, 75)]

            for step in steps:
                if step in p25_qdict:
                    p25_q.append(sum(p25_qdict[step]) / len(p25_qdict[step]))
                    p50_q.append(sum(p50_qdict[step]) / len(p50_qdict[step]))
                    p75_q.append(sum(p75_qdict[step]) / len(p75_qdict[step]))
                if step in p25_hrldict:
                    p25_hrl.append(sum(p25_hrldict[step]) / len(p25_hrldict[step]))
                    p50_hrl.append(sum(p50_hrldict[step]) / len(p50_hrldict[step]))
                    p75_hrl.append(sum(p75_hrldict[step]) / len(p75_hrldict[step]))
                if step in p25dict:
                    p25.append(sum(p25dict[step]) / len(p25dict[step]))
                    p50.append(sum(p50dict[step]) / len(p50dict[step]))
                    p75.append(sum(p75dict[step]) / len(p75dict[step]))
                if step in p25sdict:
                    p25s.append(sum(p25sdict[step]) / len(p25sdict[step]))
                    p50s.append(sum(p50sdict[step]) / len(p50sdict[step]))
                    p75s.append(sum(p75sdict[step]) / len(p75sdict[step]))
                if step in p25_dqndict:
                    p25_dqn.append(sum(p25_dqndict[step]) / len(p25_dqndict[step]))
                    p50_dqn.append(sum(p50_dqndict[step]) / len(p50_dqndict[step]))
                    p75_dqn.append(sum(p75_dqndict[step]) / len(p75_dqndict[step]))

            fig, ax = plt.subplots()
            fig.set_figheight(6)
            fig.set_figwidth(8)

            if algorithm == "jirprpni" or algorithm == "all":
                p25 = smooth(p25, 5)
                p50 = smooth(p50, 5)
                p75 = smooth(p75, 5)

                steps = np.linspace(0, (len(p25) - 1) * step_unit, len(p25), endpoint=True)
                plt.xlim(0, (len(p25) - 1) * step_unit)

                ax.plot(steps, p25, alpha=0)
                ax.plot(steps, p50, color='black', label='JIRP RPNI')
                ax.plot(steps, p75, alpha=0)

                plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
                plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

            if algorithm == "jirpsat" or algorithm == "all":
                p25s = smooth(p25s, 5)
                p50s = smooth(p50s, 5)
                p75s = smooth(p75s, 5)

                steps = np.linspace(0, (len(p25s) - 1) * step_unit, len(p25s), endpoint=True)
                plt.xlim(0, (len(p25s) - 1) * step_unit)

                ax.plot(steps, p25s, alpha=0)
                ax.plot(steps, p50s, color='green', label='JIRP SAT')
                ax.plot(steps, p75s, alpha=0)

                plt.fill_between(steps, p50s, p25s, color='green', alpha=0.25)
                plt.fill_between(steps, p50s, p75s, color='green', alpha=0.25)

            if algorithm == "qlearning" or algorithm == "all":
                p25_q = smooth(p25_q, 5)
                p50_q = smooth(p50_q, 5)
                p75_q = smooth(p75_q, 5)

                steps = np.linspace(0, (len(p25_q) - 1) * step_unit, len(p25_q), endpoint=True)
                plt.xlim(0, (len(p25_q) - 1) * step_unit)

                ax.plot(steps, p25_q, alpha=0)
                ax.plot(steps, p50_q, color='red', label='QAS')
                ax.plot(steps, p75_q, alpha=0)

                plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
                plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

            if algorithm == "ddqn" or algorithm == "all":
                p25_dqn = smooth(p25_dqn, 5)
                p50_dqn = smooth(p50_dqn, 5)
                p75_dqn = smooth(p75_dqn, 5)

                steps = np.linspace(0, (len(p25_dqn) - 1) * step_unit, len(p25_dqn), endpoint=True)
                plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

                ax.plot(steps, p25_dqn, alpha=0)
                ax.plot(steps, p50_dqn, color='purple', label='D-DQN')
                ax.plot(steps, p75_dqn, alpha=0)

                plt.fill_between(steps, p50_dqn, p25_dqn, color='purple', alpha=0.25)
                plt.fill_between(steps, p50_dqn, p75_dqn, color='purple', alpha=0.25)

            if algorithm == "hrl" or algorithm == "all":
                p25_hrl = smooth(p25_hrl, 5)
                p50_hrl = smooth(p50_hrl, 5)
                p75_hrl = smooth(p75_hrl, 5)

                steps = np.linspace(0, (len(p25_hrl) - 1) * step_unit, len(p25_hrl), endpoint=True)
                plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

                ax.plot(steps, p25_hrl, alpha=0)
                ax.plot(steps, p50_hrl, color='blue', label='HRL')
                ax.plot(steps, p75_hrl, alpha=0)

                plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
                plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

            ax.grid()



            ax.set_xlabel('number of training steps', fontsize=22)
            ax.set_ylabel('reward', fontsize=22)
            plt.ylim(-0.1, 1.1)
            plt.xlim(0, max_step)

            plt.locator_params(axis='x', nbins=5)

            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.gcf().subplots_adjust(bottom=0.15)
            plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '','','D-DQN','', '', 'HRL', ''))
            plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

            ax.tick_params(axis='both', which='major', labelsize=22)

            plt.savefig('../plotdata/figure.png', dpi=600)
            plt.show()


def export_results_office_world(task_id, algorithm):
    files = os.listdir("../plotdata/")

    step_unit = get_params_office_world('../experiments/office/tests/ground_truth.txt')[0].num_steps
    max_step = get_params_office_world('../experiments/office/tests/ground_truth.txt')[3].total_steps

    steps = np.linspace(0, max_step, (max_step / step_unit) + 1, endpoint=True)

    if task_id>0:
        p25 = [0]
        p50 = [0]
        p75 = [0]
        p25s = [0]
        p50s = [0]
        p75s = [0]
        p25_q = [0]
        p50_q = [0]
        p75_q = [0]
        p25_hrl = [0]
        p50_hrl = [0]
        p75_hrl = [0]
        p25_dqn = [0]
        p50_dqn = [0]
        p75_dqn = [0]
        files_of_interest = list()
        for file in files:
            if (("office" in file) and (".csv" in file) and (str(task_id) in file)):
                files_of_interest.append(file)

        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            if 'qlearning' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_q.append(np.percentile(row, 25))
                        p50_q.append(np.percentile(row, 50))
                        p75_q.append(np.percentile(row, 75))
            elif 'hrl' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_hrl.append(np.percentile(row, 25))
                        p50_hrl.append(np.percentile(row, 50))
                        p75_hrl.append(np.percentile(row, 75))
            elif 'dqn' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_dqn.append(np.percentile(row, 25))
                        p50_dqn.append(np.percentile(row, 50))
                        p75_dqn.append(np.percentile(row, 75))
            elif 'rpni' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25.append(np.percentile(row, 25))
                        p50.append(np.percentile(row, 50))
                        p75.append(np.percentile(row, 75))
            elif 'sat' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25s.append(np.percentile(row, 25))
                        p50s.append(np.percentile(row, 50))
                        p75s.append(np.percentile(row, 75))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)

        if algorithm == "jirprpni" or algorithm == "all":
            p25 = smooth(p25, 5)
            p50 = smooth(p50, 5)
            p75 = smooth(p75, 5)

            steps = np.linspace(0, (len(p25)-1) * step_unit, len(p25), endpoint=True)
            plt.xlim(0, (len(p25)-1) * step_unit)

            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP RPNI')
            ax.plot(steps, p75, alpha=0)

            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "jirpsat" or algorithm == "all":
            p25s = smooth(p25s, 5)
            p50s = smooth(p50s, 5)
            p75s = smooth(p75s, 5)

            steps = np.linspace(0, (len(p25s)-1) * step_unit, len(p25s), endpoint=True)
            plt.xlim(0, (len(p25s) - 1) * step_unit)

            ax.plot(steps, p25s, alpha=0)
            ax.plot(steps, p50s, color='green', label='JIRP SAT')
            ax.plot(steps, p75s, alpha=0)

            plt.fill_between(steps, p50s, p25s, color='green', alpha=0.25)
            plt.fill_between(steps, p50s, p75s, color='green', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            p25_q = smooth(p25_q, 5)
            p50_q = smooth(p50_q, 5)
            p75_q = smooth(p75_q, 5)

            steps = np.linspace(0, (len(p25_q)-1) * step_unit, len(p25_q), endpoint=True)
            plt.xlim(0, (len(p25_q) - 1) * step_unit)

            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)

            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            p25_hrl = smooth(p25_hrl, 5)
            p50_hrl = smooth(p50_hrl, 5)
            p75_hrl = smooth(p75_hrl, 5)

            steps = np.linspace(0, (len(p25_hrl)-1) * step_unit, len(p25_hrl), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        if algorithm == "ddqn" or algorithm == "all":
            p25_dqn = smooth(p25_dqn, 5)
            p50_dqn = smooth(p50_dqn, 5)
            p75_dqn = smooth(p75_dqn, 5)

            steps = np.linspace(0, (len(p25_dqn)-1) * step_unit, len(p25_dqn), endpoint=True)
            plt.xlim(0, (len(p25_dqn)-1) * step_unit)

            ax.plot(steps, p25_dqn, alpha=0)
            ax.plot(steps, p50_dqn, color='purple', label='D-DQN')
            ax.plot(steps, p75_dqn, alpha=0)

            plt.fill_between(steps, p50_dqn, p25_dqn, color='purple', alpha=0.25)
            plt.fill_between(steps, p50_dqn, p75_dqn, color='purple', alpha=0.25)

        ax.grid()


        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '', '','D-DQN','','', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

    else:

        step = 0

        p25dict = dict()
        p50dict = dict()
        p75dict = dict()
        p25sdict = dict()
        p50sdict = dict()
        p75sdict = dict()
        p25_qdict = dict()
        p50_qdict = dict()
        p75_qdict = dict()
        p25_hrldict = dict()
        p50_hrldict = dict()
        p75_hrldict = dict()
        p25_dqndict = dict()
        p50_dqndict = dict()
        p75_dqndict = dict()

        p25 = list()
        p50 = list()
        p75 = list()
        p25s = list()
        p50s = list()
        p75s = list()
        p25_q = list()
        p50_q = list()
        p75_q = list()
        p25_hrl = list()
        p50_hrl = list()
        p75_hrl = list()
        p25_dqn = list()
        p50_dqn = list()
        p75_dqn = list()

        p25dict[0] = [0,0,0,0]
        p50dict[0] = [0,0,0,0]
        p75dict[0] = [0,0,0,0]
        p25sdict[0] = [0,0,0,0]
        p50sdict[0] = [0,0,0,0]
        p75sdict[0] = [0,0,0,0]
        p25_qdict[0] = [0,0,0,0]
        p50_qdict[0] = [0,0,0,0]
        p75_qdict[0] = [0,0,0,0]
        p25_hrldict[0] = [0,0,0,0]
        p50_hrldict[0] = [0,0,0,0]
        p75_hrldict[0] = [0,0,0,0]
        p25_dqndict[0] = [0,0,0,0]
        p50_dqndict[0] = [0,0,0,0]
        p75_dqndict[0] = [0,0,0,0]

        files_dict = dict()
        for file in files:
            if (("office" in file) and (".csv" in file)):
                if "1" in file:
                    task = 1
                if "2" in file:
                    task = 2
                if "3" in file:
                    task = 3
                if "4" in file:
                    task = 4

                if task not in files_dict:
                    files_dict[task] = [file]
                else:
                    files_dict[task].append(file)

        for task in files_dict:
            for file in files_dict[task]:
                file_str = ("../plotdata/") + file
                if 'qlearn' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)

                        for row_ in readcsv:
                            if len(row_) > 1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_qdict:
                                p25_qdict[step].append(np.percentile(row, 25))
                                p50_qdict[step].append(np.percentile(row, 50))
                                p75_qdict[step].append(np.percentile(row, 75))
                            else:
                                p25_qdict[step] = [np.percentile(row, 25)]
                                p50_qdict[step] = [np.percentile(row, 50)]
                                p75_qdict[step] = [np.percentile(row, 75)]

                elif 'hrl' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_) > 1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_hrldict:
                                p25_hrldict[step].append(np.percentile(row, 25))
                                p50_hrldict[step].append(np.percentile(row, 50))
                                p75_hrldict[step].append(np.percentile(row, 75))
                            else:
                                p25_hrldict[step] = [np.percentile(row, 25)]
                                p50_hrldict[step] = [np.percentile(row, 50)]
                                p75_hrldict[step] = [np.percentile(row, 75)]

                elif 'dqn' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_) > 1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_dqndict:
                                p25_dqndict[step].append(np.percentile(row, 25))
                                p50_dqndict[step].append(np.percentile(row, 50))
                                p75_dqndict[step].append(np.percentile(row, 75))
                            else:
                                p25_dqndict[step] = [np.percentile(row, 25)]
                                p50_dqndict[step] = [np.percentile(row, 50)]
                                p75_dqndict[step] = [np.percentile(row, 75)]

                elif 'rpni' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_) > 1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25dict:
                                p25dict[step].append(np.percentile(row, 25))
                                p50dict[step].append(np.percentile(row, 50))
                                p75dict[step].append(np.percentile(row, 75))
                            else:
                                p25dict[step] = [np.percentile(row, 25)]
                                p50dict[step] = [np.percentile(row, 50)]
                                p75dict[step] = [np.percentile(row, 75)]
                elif 'sat' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_) > 1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25sdict:
                                p25sdict[step].append(np.percentile(row, 25))
                                p50sdict[step].append(np.percentile(row, 50))
                                p75sdict[step].append(np.percentile(row, 75))
                            else:
                                p25sdict[step] = [np.percentile(row, 25)]
                                p50sdict[step] = [np.percentile(row, 50)]
                                p75sdict[step] = [np.percentile(row, 75)]

        for step in steps:
            if step in p25_qdict:
                p25_q.append(sum(p25_qdict[step])/len(p25_qdict[step]))
                p50_q.append(sum(p50_qdict[step])/len(p50_qdict[step]))
                p75_q.append(sum(p75_qdict[step])/len(p75_qdict[step]))
            if step in p25_hrldict:
                p25_hrl.append(sum(p25_hrldict[step])/len(p25_hrldict[step]))
                p50_hrl.append(sum(p50_hrldict[step])/len(p50_hrldict[step]))
                p75_hrl.append(sum(p75_hrldict[step])/len(p75_hrldict[step]))
            if step in p25dict:
                p25.append(sum(p25dict[step])/len(p25dict[step]))
                p50.append(sum(p50dict[step])/len(p50dict[step]))
                p75.append(sum(p75dict[step])/len(p75dict[step]))
            if step in p25sdict:
                p25s.append(sum(p25sdict[step])/len(p25sdict[step]))
                p50s.append(sum(p50sdict[step])/len(p50sdict[step]))
                p75s.append(sum(p75sdict[step])/len(p75sdict[step]))
            if step in p25_dqndict:
                p25_dqn.append(sum(p25_dqndict[step]) / len(p25_dqndict[step]))
                p50_dqn.append(sum(p50_dqndict[step]) / len(p50_dqndict[step]))
                p75_dqn.append(sum(p75_dqndict[step]) / len(p75_dqndict[step]))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)

        if algorithm == "jirprpni" or algorithm == "all":
            p25 = smooth(p25, 5)
            p50 = smooth(p50, 5)
            p75 = smooth(p75, 5)

            steps = np.linspace(0, (len(p25) - 1) * step_unit, len(p25), endpoint=True)
            plt.xlim(0, (len(p25) - 1) * step_unit)

            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP RPNI')
            ax.plot(steps, p75, alpha=0)

            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "jirpsat" or algorithm == "all":
            p25s = smooth(p25s, 5)
            p50s = smooth(p50s, 5)
            p75s = smooth(p75s, 5)

            steps = np.linspace(0, (len(p25s) - 1) * step_unit, len(p25s), endpoint=True)
            plt.xlim(0, (len(p25s) - 1) * step_unit)

            ax.plot(steps, p25s, alpha=0)
            ax.plot(steps, p50s, color='green', label='JIRP SAT')
            ax.plot(steps, p75s, alpha=0)

            plt.fill_between(steps, p50s, p25s, color='green', alpha=0.25)
            plt.fill_between(steps, p50s, p75s, color='green', alpha=0.25)

        if algorithm == "ddqn" or algorithm == "all":
            p25_dqn = smooth(p25_dqn, 5)
            p50_dqn = smooth(p50_dqn, 5)
            p75_dqn = smooth(p75_dqn, 5)

            steps = np.linspace(0, (len(p25_dqn) - 1) * step_unit, len(p25_dqn), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_dqn, alpha=0)
            ax.plot(steps, p50_dqn, color='purple', label='D-DQN')
            ax.plot(steps, p75_dqn, alpha=0)

            plt.fill_between(steps, p50_dqn, p25_dqn, color='purple', alpha=0.25)
            plt.fill_between(steps, p50_dqn, p75_dqn, color='purple', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            p25_q = smooth(p25_q, 5)
            p50_q = smooth(p50_q, 5)
            p75_q = smooth(p75_q, 5)

            steps = np.linspace(0, (len(p25_q) - 1) * step_unit, len(p25_q), endpoint=True)
            plt.xlim(0, (len(p25_q) - 1) * step_unit)

            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)

            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            p25_hrl = smooth(p25_hrl, 5)
            p50_hrl = smooth(p50_hrl, 5)
            p75_hrl = smooth(p75_hrl, 5)

            steps = np.linspace(0, (len(p25_hrl) - 1) * step_unit, len(p25_hrl), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)
        ax.grid()


        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        if algorithm == "all":
            plt.xlim(0,max_step)
        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '', '','D-DQN','','', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.32), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()


def export_results_craft_world(task_id, algorithm):
    files = os.listdir("../plotdata/")

    step_unit = get_params_craft_world('../experiments/craft/tests/ground_truth.txt')[0].num_steps
    max_step = get_params_craft_world('../experiments/craft/tests/ground_truth.txt')[3].total_steps

    steps = np.linspace(0, max_step, (max_step / step_unit) + 1, endpoint=True)

    if task_id>0:
        p25 = [0]
        p50 = [0]
        p75 = [0]
        p25s = [0]
        p50s = [0]
        p75s = [0]
        p25_q = [0]
        p50_q = [0]
        p75_q = [0]
        p25_hrl = [0]
        p50_hrl = [0]
        p75_hrl = [0]
        p25_dqn = [0]
        p50_dqn = [0]
        p75_dqn = [0]
        files_of_interest = list()
        for file in files:
            if (("craft" in file) and (".csv" in file) and (str(task_id) in file)):
                files_of_interest.append(file)

        for file in files_of_interest:
            file_str = ("../plotdata/") + file
            if 'qlearning' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_q.append(np.percentile(row,25))
                        p50_q.append(np.percentile(row,50))
                        p75_q.append(np.percentile(row,75))
            elif 'hrl' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_hrl.append(np.percentile(row,25))
                        p50_hrl.append(np.percentile(row,50))
                        p75_hrl.append(np.percentile(row,75))
            elif 'dqn' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25_dqn.append(np.percentile(row, 25))
                        p50_dqn.append(np.percentile(row, 50))
                        p75_dqn.append(np.percentile(row, 75))
            elif 'rpni' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25.append(np.percentile(row,25))
                        p50.append(np.percentile(row,50))
                        p75.append(np.percentile(row,75))
            elif 'sat' in file:
                with open(file_str) as csvfile:
                    readcsv = csv.reader(csvfile)
                    for row_ in readcsv:
                        if len(row_) > 1:
                            row = list(map(int, row_))
                        else:
                            row = [float(row_[0])]
                        p25s.append(np.percentile(row,25))
                        p50s.append(np.percentile(row,50))
                        p75s.append(np.percentile(row,75))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)


        if algorithm == "jirprpni" or algorithm == "all":
            p25 = smooth(p25, 5)
            p50 = smooth(p50, 5)
            p75 = smooth(p75, 5)

            steps = np.linspace(0, (len(p25)-1) * step_unit, len(p25), endpoint=True)
            plt.xlim(0, (len(p25)-1) * step_unit)

            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP RPNI')
            ax.plot(steps, p75, alpha=0)

            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "jirpsat" or algorithm == "all":
            p25s = smooth(p25s, 5)
            p50s = smooth(p50s, 5)
            p75s = smooth(p75s, 5)

            steps = np.linspace(0, (len(p25s)-1) * step_unit, len(p25s), endpoint=True)
            plt.xlim(0, (len(p25s) - 1) * step_unit)

            ax.plot(steps, p25s, alpha=0)
            ax.plot(steps, p50s, color='green', label='JIRP SAT')
            ax.plot(steps, p75s, alpha=0)

            plt.fill_between(steps, p50s, p25s, color='green', alpha=0.25)
            plt.fill_between(steps, p50s, p75s, color='green', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            p25_q = smooth(p25_q, 5)
            p50_q = smooth(p50_q, 5)
            p75_q = smooth(p75_q, 5)

            steps = np.linspace(0, (len(p25_q)-1) * step_unit, len(p25_q), endpoint=True)
            plt.xlim(0, (len(p25_q) - 1) * step_unit)

            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)

            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "hrl" or algorithm == "all":
            p25_hrl = smooth(p25_hrl, 5)
            p50_hrl = smooth(p50_hrl, 5)
            p75_hrl = smooth(p75_hrl, 5)

            steps = np.linspace(0, (len(p25_hrl)-1) * step_unit, len(p25_hrl), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)

        if algorithm == "ddqn" or algorithm == "all":
            p25_dqn = smooth(p25_dqn, 5)
            p50_dqn = smooth(p50_dqn, 5)
            p75_dqn = smooth(p75_dqn, 5)

            steps = np.linspace(0, (len(p25_dqn)-1) * step_unit, len(p25_dqn), endpoint=True)
            plt.xlim(0, (len(p25_dqn)-1) * step_unit)

            ax.plot(steps, p25_dqn, alpha=0)
            ax.plot(steps, p50_dqn, color='purple', label='D-DQN')
            ax.plot(steps, p75_dqn, alpha=0)

            plt.fill_between(steps, p50_dqn, p25_dqn, color='purple', alpha=0.25)
            plt.fill_between(steps, p50_dqn, p75_dqn, color='purple', alpha=0.25)

        ax.grid()

        if algorithm == "all":
            plt.xlim(0,max_step)


        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '','','D-DQN','','', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()


    else:

        step = 0

        p25dict = dict()
        p50dict = dict()
        p75dict = dict()
        p25sdict = dict()
        p50sdict = dict()
        p75sdict = dict()
        p25_qdict = dict()
        p50_qdict = dict()
        p75_qdict = dict()
        p25_hrldict = dict()
        p50_hrldict = dict()
        p75_hrldict = dict()
        p25_dqndict = dict()
        p50_dqndict = dict()
        p75_dqndict = dict()

        p25 = list()
        p50 = list()
        p75 = list()
        p25s = list()
        p50s = list()
        p75s = list()
        p25_q = list()
        p50_q = list()
        p75_q = list()
        p25_hrl = list()
        p50_hrl = list()
        p75_hrl = list()
        p25_dqn = list()
        p50_dqn = list()
        p75_dqn = list()

        p25dict[0] = [0,0,0,0]
        p50dict[0] = [0,0,0,0]
        p75dict[0] = [0,0,0,0]
        p25sdict[0] = [0,0,0,0]
        p50sdict[0] = [0,0,0,0]
        p75sdict[0] = [0,0,0,0]
        p25_qdict[0] = [0,0,0,0]
        p50_qdict[0] = [0,0,0,0]
        p75_qdict[0] = [0,0,0,0]
        p25_hrldict[0] = [0,0,0,0]
        p50_hrldict[0] = [0,0,0,0]
        p75_hrldict[0] = [0,0,0,0]
        p25_dqndict[0] = [0,0,0,0]
        p50_dqndict[0] = [0,0,0,0]
        p75_dqndict[0] = [0,0,0,0]

        files_dict = dict()
        for file in files:
            if (("craft" in file) and (".csv" in file)):
                if "1" in file:
                    task = 1
                if "2" in file:
                    task = 2
                if "3" in file:
                    task = 3
                if "4" in file:
                    task = 4

                if task not in files_dict:
                    files_dict[task] = [file]
                else:
                    files_dict[task].append(file)

        for task in files_dict:
            for file in files_dict[task]:
                file_str = ("../plotdata/") + file
                if 'qlearning' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)

                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_qdict:
                                p25_qdict[step].append(np.percentile(row, 25))
                                p50_qdict[step].append(np.percentile(row, 50))
                                p75_qdict[step].append(np.percentile(row, 75))
                            else:
                                p25_qdict[step] = [np.percentile(row, 25)]
                                p50_qdict[step] = [np.percentile(row, 50)]
                                p75_qdict[step] = [np.percentile(row, 75)]

                elif 'hrl' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_hrldict:
                                p25_hrldict[step].append(np.percentile(row, 25))
                                p50_hrldict[step].append(np.percentile(row, 50))
                                p75_hrldict[step].append(np.percentile(row, 75))
                            else:
                                p25_hrldict[step] = [np.percentile(row, 25)]
                                p50_hrldict[step] = [np.percentile(row, 50)]
                                p75_hrldict[step] = [np.percentile(row, 75)]

                elif 'dqn' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_) > 1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25_dqndict:
                                p25_dqndict[step].append(np.percentile(row, 25))
                                p50_dqndict[step].append(np.percentile(row, 50))
                                p75_dqndict[step].append(np.percentile(row, 75))
                            else:
                                p25_dqndict[step] = [np.percentile(row, 25)]
                                p50_dqndict[step] = [np.percentile(row, 50)]
                                p75_dqndict[step] = [np.percentile(row, 75)]

                elif 'rpni' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25dict:
                                p25dict[step].append(np.percentile(row, 25))
                                p50dict[step].append(np.percentile(row, 50))
                                p75dict[step].append(np.percentile(row, 75))
                            else:
                                p25dict[step] = [np.percentile(row, 25)]
                                p50dict[step] = [np.percentile(row, 50)]
                                p75dict[step] = [np.percentile(row, 75)]
                elif 'sat' in file:
                    with open(file_str) as csvfile:
                        step = 0
                        readcsv = csv.reader(csvfile)
                        for row_ in readcsv:
                            if len(row_)>1:
                                row = list(map(int, row_))
                            else:
                                row = [float(row_[0])]
                            step += step_unit
                            if step in p25sdict:
                                p25sdict[step].append(np.percentile(row, 25))
                                p50sdict[step].append(np.percentile(row, 50))
                                p75sdict[step].append(np.percentile(row, 75))
                            else:
                                p25sdict[step] = [np.percentile(row, 25)]
                                p50sdict[step] = [np.percentile(row, 50)]
                                p75sdict[step] = [np.percentile(row, 75)]


        for step in steps:
            if step in p25_qdict:
                p25_q.append(sum(p25_qdict[step])/len(p25_qdict[step]))
                p50_q.append(sum(p50_qdict[step])/len(p50_qdict[step]))
                p75_q.append(sum(p75_qdict[step])/len(p75_qdict[step]))
            if step in p25_hrldict:
                p25_hrl.append(sum(p25_hrldict[step])/len(p25_hrldict[step]))
                p50_hrl.append(sum(p50_hrldict[step])/len(p50_hrldict[step]))
                p75_hrl.append(sum(p75_hrldict[step])/len(p75_hrldict[step]))
            if step in p25dict:
                p25.append(sum(p25dict[step])/len(p25dict[step]))
                p50.append(sum(p50dict[step])/len(p50dict[step]))
                p75.append(sum(p75dict[step])/len(p75dict[step]))
            if step in p25sdict:
                p25s.append(sum(p25sdict[step])/len(p25sdict[step]))
                p50s.append(sum(p50sdict[step])/len(p50sdict[step]))
                p75s.append(sum(p75sdict[step])/len(p75sdict[step]))
            if step in p25_dqndict:
                p25_dqn.append(sum(p25_dqndict[step]) / len(p25_dqndict[step]))
                p50_dqn.append(sum(p50_dqndict[step]) / len(p50_dqndict[step]))
                p75_dqn.append(sum(p75_dqndict[step]) / len(p75_dqndict[step]))

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(8)

        if algorithm=="jirprpni" or algorithm=="all":
            p25 = smooth(p25,5)
            p50 = smooth(p50,5)
            p75 = smooth(p75,5)

            steps = np.linspace(0, (len(p25)-1) * step_unit, len(p25), endpoint=True)
            plt.xlim(0, (len(p25)-1) * step_unit)

            ax.plot(steps, p25, alpha=0)
            ax.plot(steps, p50, color='black', label='JIRP RPNI')
            ax.plot(steps, p75, alpha=0)

            plt.fill_between(steps, p50, p25, color='black', alpha=0.25)
            plt.fill_between(steps, p50, p75, color='black', alpha=0.25)

        if algorithm == "jirpsat" or algorithm == "all":
            p25s = smooth(p25s, 5)
            p50s = smooth(p50s, 5)
            p75s = smooth(p75s, 5)

            steps = np.linspace(0, (len(p25s)-1) * step_unit, len(p25s), endpoint=True)
            plt.xlim(0, (len(p25s) - 1) * step_unit)

            ax.plot(steps, p25s, alpha=0)
            ax.plot(steps, p50s, color='green', label='JIRP SAT')
            ax.plot(steps, p75s, alpha=0)

            plt.fill_between(steps, p50s, p25s, color='green', alpha=0.25)
            plt.fill_between(steps, p50s, p75s, color='green', alpha=0.25)

        if algorithm == "qlearning" or algorithm == "all":
            p25_q = smooth(p25_q, 5)
            p50_q = smooth(p50_q, 5)
            p75_q = smooth(p75_q, 5)

            steps = np.linspace(0, (len(p25_q)-1) * step_unit, len(p25_q), endpoint=True)
            plt.xlim(0, (len(p25_q) - 1) * step_unit)

            ax.plot(steps, p25_q, alpha=0)
            ax.plot(steps, p50_q, color='red', label='QAS')
            ax.plot(steps, p75_q, alpha=0)

            plt.fill_between(steps, p50_q, p25_q, color='red', alpha=0.25)
            plt.fill_between(steps, p50_q, p75_q, color='red', alpha=0.25)

        if algorithm == "ddqn" or algorithm == "all":
            p25_dqn = smooth(p25_dqn, 5)
            p50_dqn = smooth(p50_dqn, 5)
            p75_dqn = smooth(p75_dqn, 5)

            steps = np.linspace(0, (len(p25_dqn) - 1) * step_unit, len(p25_dqn), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_dqn, alpha=0)
            ax.plot(steps, p50_dqn, color='purple', label='D-DQN')
            ax.plot(steps, p75_dqn, alpha=0)

            plt.fill_between(steps, p50_dqn, p25_dqn, color='purple', alpha=0.25)
            plt.fill_between(steps, p50_dqn, p75_dqn, color='purple', alpha=0.25)


        if algorithm == "hrl" or algorithm == "all":
            p25_hrl = smooth(p25_hrl, 5)
            p50_hrl = smooth(p50_hrl, 5)
            p75_hrl = smooth(p75_hrl, 5)

            steps = np.linspace(0, (len(p25_hrl)-1) * step_unit, len(p25_hrl), endpoint=True)
            plt.xlim(0, (len(p25_hrl) - 1) * step_unit)

            ax.plot(steps, p25_hrl, alpha=0)
            ax.plot(steps, p50_hrl, color='blue', label='HRL')
            ax.plot(steps, p75_hrl, alpha=0)

            plt.fill_between(steps, p50_hrl, p25_hrl, color='blue', alpha=0.25)
            plt.fill_between(steps, p50_hrl, p75_hrl, color='blue', alpha=0.25)
        ax.grid()

        ax.set_xlabel('number of training steps', fontsize=22)
        ax.set_ylabel('reward', fontsize=22)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, max_step)

        plt.locator_params(axis='x', nbins=5)

        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gca().legend(('', 'JIRP RPNI', '', '', 'JIRP SAT', '', '', 'QAS', '','','D-DQN','', '', 'HRL', ''))
        plt.legend(loc='upper right', bbox_to_anchor=(1, 0.8), prop={'size': 14})

        ax.tick_params(axis='both', which='major', labelsize=22)

        plt.savefig('../plotdata/figure.png', dpi=600)
        plt.show()

if __name__ == "__main__":


    # EXAMPLE: python3 export_summary.py --world="craft"

    # Getting params
    worlds     = ["office", "craft", "traffic"]
    algorithms = ["jirpsat", "jirprpni", "qlearning", "hrl", "ddqn","all"]

    print("Note: ensure that runs correspond with current parameters for curriculum.total_steps and testing_params.num_steps!")
    print("")

    parser = argparse.ArgumentParser(prog="export_summary", description='After running the experiments, this algorithm computes a summary of the results.')
    parser.add_argument('--world', default='traffic', type=str,
                        help='This parameter indicated which world to solve.')
    parser.add_argument('--algorithm', default='jirpsat', type=str,
                        help='This parameter indicated which algorithm to solve. Set to "all" to graph all methods.')
    parser.add_argument('--task', default=1, type=int,
                        help='This parameter indicates which task to display. Set to zero to graph all tasks.')

    args = parser.parse_args()
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")



    # Computing the experiment summary
    world = args.world
    if world == "office":
        export_results_office_world(args.task, args.algorithm)
    if world == "craft":
        export_results_craft_world(args.task, args.algorithm)
    if world == "traffic":
        export_results_traffic_world(args.task, args.algorithm)
